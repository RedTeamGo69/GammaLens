# =============================================================================
# gap_experiment.py
# Walk-forward scoring of candidate regressors against the production spec.
#
# The question this answers with data instead of argument:
#   "Does the Friday-close -> Monday-open gap (and the other candidates in
#    candidate_features.py) predict the week's range beyond what M3_extended
#    already knows?"
#
# Every arm is the baseline spec PLUS one candidate group, scored through
# walkforward.compare_configs with the EXISTING adoption gate — no new
# thresholds are invented here. An arm either clears _MIN_R2_IMPROVEMENT /
# _MIN_MAPE_IMPROVEMENT without worsening coverage by more than 2pp, or it
# is rejected with its numbers on the record.
#
# ---------------------------------------------------------------------------
# WHY THE GAP IS A LEGAL REGRESSOR
# ---------------------------------------------------------------------------
# Entry is Monday-open or later, and the target window starts at Monday's
# open (range_pct = (H-L)/spx_open). The
# weekend gap therefore ends exactly where the target begins: it is observable
# at the decision (no look-ahead) and disjoint from the forecast quantity (no
# overlapping-window R2 inflation). See candidate_features.weekend_gap.
#
# It also needs no ES futures data. The observable quantity is
# spx_open[t]/spx_close[t-1] - 1, straight off weekly_spx. ES would only add
# the overnight PATH (Sun 18:00 - Mon 09:30 high-low), which carries quarterly
# roll artifacts and absent holiday sessions — worth doing only if the cash
# gap first shows signal here.
#
# READ-ONLY: only SELECTs through the existing loaders. Nothing is written.
#
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.gap_experiment
#   ... [--ticker SPX] [--spec M3_extended] [--step 4] [--min-train 104]
# =============================================================================

import argparse
import logging
import sys

import pandas as pd

from range_finder.candidate_features import build_candidate_matrix
from range_finder.har_model import (
    MODEL_SPECS,
    TRAIN_WINDOW_YEARS,
    feature_has_enough_data,
)
from range_finder.walkforward import WalkForwardConfig, compare_configs

log = logging.getLogger(__name__)

# Candidate groups, each scored as "baseline + this group". Grouped rather
# than one-column-at-a-time because the members of a group are the two halves
# of a single decomposition — splitting variance into overnight/intraday and
# then testing only one half answers nothing.
CANDIDATE_ARMS: dict[str, tuple[str, ...]] = {
    "+ weekend gap (abs)":      ("abs_weekend_gap",),
    "+ weekend gap (signed)":   ("weekend_gap",),
    "+ overnight/intraday RV":  ("rv_overnight", "rv_intraday"),
    "+ overnight share":        ("overnight_share",),
    "+ vix9d level":            ("vix9d_close",),
    "+ semivariance":           ("rs_neg", "rs_pos"),
    "+ HARQ (har_d1 x sqrtRQ)": ("harq_d1",),
}


def _load_daily_ohlc(ticker: str, years: float) -> pd.DataFrame:
    """Daily OHLCV for the ticker's feature source, full bar contract.

    feature_builder.fetch_daily_underlying discards everything but `close`,
    which is enough for HV but not for the overnight/intraday split — that
    needs `open` too. bar_sources.fetch_daily_bars already returns the full
    open/high/low/close/volume contract, so this goes straight to it.
    """
    from phase1.ticker_config import feature_source_ticker, get_config
    from range_finder.bar_sources import fetch_daily_bars

    src = feature_source_ticker(ticker)
    yf_symbol = "^GSPC" if src == "SPX" else get_config(src)["yf_symbol"]
    return fetch_daily_bars(yf_symbol, src, years, src)


def build_scored_frame(
    conn,
    ticker: str = "SPX",
    spec: str = "M3_extended",
    years: float = TRAIN_WINDOW_YEARS,
) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix joined with every candidate column, plus the usable
    baseline spec columns. Read-only."""
    from range_finder.feature_builder import _load_weekly_for_ticker, get_features

    feats = get_features(conn, exclude_covid=True, ticker=ticker).sort_index()
    if feats.empty:
        raise RuntimeError(f"model_features empty for {ticker} — bootstrap first")

    weekly = _load_weekly_for_ticker(conn, ticker).sort_index()
    daily = _load_daily_ohlc(ticker, years)

    candidates = build_candidate_matrix(weekly, daily)
    df = feats.join(candidates, how="left")

    # HARQ is an interaction, not a standalone column: sqrt(RQ) scales the
    # most recent lag's coefficient (Bollerslev/Patton/Quaedvlieg 2016).
    if "rq_sqrt" in df.columns and "har_d1" in df.columns:
        df["harq_d1"] = df["har_d1"] * df["rq_sqrt"]

    base_cols = [c for c in MODEL_SPECS[spec] if feature_has_enough_data(df, c)]
    if len(base_cols) < 2:
        raise RuntimeError(f"{spec}: too few usable baseline features {base_cols}")

    return df, base_cols


def align_common_sample(
    df: pd.DataFrame,
    base_cols: list[str],
    arms: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Restrict every arm to one identical row set.

    walk_forward_evaluate drops NaN rows per-config, so an arm carrying a
    sparser candidate would be scored on a DIFFERENT (and usually shorter,
    more recent) sample than the baseline. delta_r2 across two samples is not
    a comparison, and the direction of the bias is unknowable. Dropping to the
    intersection up front costs rows but makes every delta honest.

    Arms whose candidates are entirely absent are dropped and reported, so a
    missing data source shows up as a named omission rather than a silently
    smaller table.
    """
    live_arms = {}
    for label, cols in arms.items():
        present = [c for c in cols if c in df.columns and df[c].notna().any()]
        if len(present) == len(cols):
            live_arms[label] = cols
        else:
            missing = [c for c in cols if c not in present]
            log.warning(f"dropping arm '{label}' — no data for {missing}")

    needed = list(base_cols) + ["log_range"]
    for cols in live_arms.values():
        needed.extend(cols)
    needed = list(dict.fromkeys(needed))

    n_before = int(df[list(base_cols) + ["log_range"]].dropna().shape[0])
    aligned = df.dropna(subset=needed)
    n_after = len(aligned)

    log.info(
        f"common sample: {n_after} rows "
        f"({aligned.index.min().date()} -> {aligned.index.max().date()}); "
        f"baseline alone would have had {n_before} "
        f"({n_before - n_after} rows cost by candidate coverage)"
    )
    return aligned, live_arms


def run_experiment(
    conn,
    ticker: str = "SPX",
    spec: str = "M3_extended",
    min_train: int = 104,
    step: int = 4,
) -> pd.DataFrame:
    """Score every candidate arm against the baseline. Returns the comparison
    frame from compare_configs (with its `keep` column)."""
    df, base_cols = build_scored_frame(conn, ticker=ticker, spec=spec)
    aligned, live_arms = align_common_sample(df, base_cols, CANDIDATE_ARMS)

    baseline_label = f"{spec} (baseline)"
    configs = [WalkForwardConfig(label=baseline_label,
                                 feature_cols=tuple(base_cols))]
    for label, cols in live_arms.items():
        configs.append(WalkForwardConfig(
            label=label,
            feature_cols=tuple(base_cols) + tuple(cols),
        ))

    log.info(f"{spec} baseline features: {base_cols}")
    log.info(f"scoring {len(configs) - 1} candidate arms on {len(aligned)} rows")

    return compare_configs(aligned, configs, baseline_label=baseline_label,
                           min_train=min_train, step=step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--spec", default="M3_extended")
    parser.add_argument("--min-train", type=int, default=104)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    # spread_levels' import-time basicConfig(INFO) wins the root config race —
    # override it so hundreds of walk-forward fits don't print their summaries.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("range_finder.har_model").setLevel(logging.WARNING)

    # No DATABASE_URL pre-check here: db.get_connection resolves it from
    # Streamlit secrets OR the environment, and raises its own clear message
    # when neither has it. An env-only guard would refuse runs that work.
    from range_finder.db import get_connection

    conn = get_connection()
    comp = run_experiment(conn, ticker=args.ticker, spec=args.spec,
                          min_train=args.min_train, step=args.step)

    kept = comp[comp["keep"]]["label"].tolist() if "keep" in comp.columns else []
    print("\nVERDICT")
    if kept:
        for label in kept:
            print(f"  CLEARS GATE : {label}")
        print("\n  A cleared arm is a candidate for promotion into "
              "feature_builder + MODEL_SPECS. Promotion must carry the lag "
              "class from candidate_features (same-week columns join with NO "
              "lag; prior-week columns go through lag_for_join).")
    else:
        print("  No arm cleared the adoption gate — the baseline spec stands.")


if __name__ == "__main__":
    sys.exit(main())
