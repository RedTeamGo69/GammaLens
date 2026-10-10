# =============================================================================
# candidate_features.py
# Candidate regressors for the weekly range model — NOT production features.
#
# Everything here is a pure frame->frame transform over data the existing
# loaders already return (data_collector.get_weekly_spx, bar_sources.
# fetch_daily_bars). Nothing writes to the DB and nothing is joined into
# model_features: these columns exist so gap_experiment.py can score them
# through the walk-forward adoption gate BEFORE any of them earn a place in
# feature_builder.build_features and the model_features schema.
#
# ---------------------------------------------------------------------------
# THE LAG BOUNDARY — read this before adding a function
# ---------------------------------------------------------------------------
# The weekly target is range_pct = (H-L)/spx_open over Monday-open ->
# Friday-close (data_collector.py), and entry is Monday-open or later. So a
# candidate is legal at week t if it is observable by Monday's open of week
# t. Two disjoint classes:
#
#   SAME-WEEK (no lag): quantities fixed AT Monday's open of week t. The
#     weekend gap is the only one here — it ends exactly where the target
#     window begins, so it overlaps neither the target nor the future.
#     Joined at week t directly, mirroring how GEX is joined
#     (feature_builder.py: "Monday open — same week, no lag needed").
#
#   PRIOR-WEEK (lag 1): path statistics accumulated over week t-1 and only
#     complete at its Friday close — semivariance, quarticity, the
#     overnight/intraday variance split. These MUST go through
#     lag_for_join() or they leak the target week's own path into its
#     forecast.
#
# Every public function below documents which class it returns. Mixing them
# up is the one mistake in this module that a green test suite would not
# catch, because both produce a plausible in-sample R2.
# =============================================================================

import logging

import numpy as np
import pandas as pd

# Reuse the production lag helper rather than re-deriving shift(1, freq="7D").
# Beyond value-for-value equality it also emits an index entry one week past
# the frame's last row, which is what lets an appended forecast-row scaffold
# pick up lagged features instead of landing on NaN.
from range_finder.feature_builder import _lag_one_week

log = logging.getLogger(__name__)

# A weekly frame's rows must be exactly 7 days apart for a "previous row" to
# mean "previous week". Any wider spacing means a week is missing from the
# frame (a filtered/subset frame, e.g. exclude_covid), and differencing across
# that hole would manufacture a multi-week "gap" that never traded as one.
_WEEK = pd.Timedelta(days=7)


# =============================================================================
# SAME-WEEK CANDIDATES (no lag — observable at Monday's open of week t)
# =============================================================================

def weekend_gap(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Friday-close -> Monday-open return, attributed to the week it opens.

    THIS IS A SAME-WEEK FEATURE — join it at week t with NO lag.

    gap[t] = spx_open[t] / spx_close[t-1] - 1

    The target window for week t starts at spx_open[t], so this move sits
    strictly outside the quantity being forecast and strictly before the
    Monday-open decision. That is what makes it legal: it is neither a
    component of the target (no overlapping-window R2 inflation) nor
    unobservable at entry (no look-ahead).

    Holiday Mondays are kept deliberately. When Monday is a holiday the
    weekly bar opens Tuesday and the move spans ~4 calendar days rather than
    ~2 — but it is still exactly "the move from the last observable close to
    the anchor open", which is the quantity the model needs. Only genuinely
    MISSING weeks (index spacing > 7 days, e.g. a COVID-filtered frame) are
    nulled, because differencing across those would span untraded weeks.

    Returns a Monday-indexed frame with:
        weekend_gap      signed return
        abs_weekend_gap  magnitude (the vol-relevant form)
    """
    out = pd.DataFrame(index=weekly_df.index.copy())
    out.index.name = weekly_df.index.name or "week_start"

    if weekly_df.empty or not {"spx_open", "spx_close"} <= set(weekly_df.columns):
        out["weekend_gap"] = np.nan
        out["abs_weekend_gap"] = np.nan
        return out

    df = weekly_df.sort_index()
    prior_close = df["spx_close"].shift(1)

    gap = df["spx_open"] / prior_close - 1.0

    # Null any row whose predecessor is not exactly one week back.
    spacing = df.index.to_series().diff()
    gap = gap.where(spacing == _WEEK)

    # A non-positive price is bad data, not a -100% gap.
    gap = gap.where((df["spx_open"] > 0) & (prior_close > 0))

    out["weekend_gap"] = gap.reindex(out.index)
    out["abs_weekend_gap"] = out["weekend_gap"].abs()
    return out


# =============================================================================
# PRIOR-WEEK CANDIDATES (lag 1 — complete only at week t-1's Friday close)
# =============================================================================

def _daily_log_returns(daily_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Overnight / intraday / close-to-close log returns from a daily OHLC
    frame (bar_sources output contract: open/high/low/close/volume)."""
    df = daily_ohlc.sort_index()
    prior_close = df["close"].shift(1)

    valid = (df["open"] > 0) & (df["close"] > 0) & (prior_close > 0)

    out = pd.DataFrame(index=df.index)
    out["r_overnight"] = np.log(df["open"] / prior_close).where(valid)
    out["r_intraday"] = np.log(df["close"] / df["open"]).where(valid)
    out["r_close"] = np.log(df["close"] / prior_close).where(valid)
    return out


def _to_monday_weekly(frame: pd.DataFrame, how: str = "sum") -> pd.DataFrame:
    """Resample a daily frame to weekly and relabel Friday -> that week's
    Monday, matching feature_builder.compute_hv_windows' convention so the
    result joins straight onto the weekly feature matrix."""
    weekly = getattr(frame.resample("W-FRI"), how)()
    weekly.index = weekly.index - pd.offsets.Week(weekday=0)
    weekly.index.name = "week_start"
    return weekly


def overnight_intraday_split(daily_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Weekly realized variance split into its overnight and intraday parts.

    PRIOR-WEEK — pass through lag_for_join() before joining.

    Close-to-close realized variance silently blends two processes with very
    different persistence: the intraday session and the gaps between sessions.
    Separating them is the honest, fully-observable version of "add ES
    overnight moves as a feature" — it needs no futures feed, no
    continuous-contract roll adjustment, and no Sunday-session handling,
    because the cash open/close already brackets every gap.

    Returns Monday-indexed:
        rv_overnight     sum of squared overnight log returns
        rv_intraday      sum of squared intraday log returns
        overnight_share  rv_overnight / (rv_overnight + rv_intraday)
    """
    empty = pd.DataFrame(columns=["rv_overnight", "rv_intraday", "overnight_share"])
    if daily_ohlc.empty or not {"open", "close"} <= set(daily_ohlc.columns):
        return empty

    rets = _daily_log_returns(daily_ohlc)
    sq = pd.DataFrame({
        "rv_overnight": rets["r_overnight"] ** 2,
        "rv_intraday": rets["r_intraday"] ** 2,
    })

    weekly = _to_monday_weekly(sq, how="sum")
    total = weekly["rv_overnight"] + weekly["rv_intraday"]
    weekly["overnight_share"] = (weekly["rv_overnight"] / total).where(total > 0)
    return weekly


def semivariance_split(daily_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Weekly realized semivariance, signed (Patton & Sheppard 2015).

    PRIOR-WEEK — pass through lag_for_join() before joining.

    Downside semivariance is the component that drives future volatility;
    pooling both signs into one RV throws that asymmetry away. This is the
    candidate that could give the bands a genuinely asymmetric shape, which
    estimate_side_share_quantile cannot express — it pools s_up with
    1 - s_up and is direction-neutral by construction (har_model.py).

    Returns Monday-indexed:
        rs_neg      sum of squared negative close-to-close log returns
        rs_pos      sum of squared positive close-to-close log returns
        rs_skew     (rs_pos - rs_neg) / (rs_pos + rs_neg)
    """
    empty = pd.DataFrame(columns=["rs_neg", "rs_pos", "rs_skew"])
    if daily_ohlc.empty or "close" not in daily_ohlc.columns:
        return empty

    r = _daily_log_returns(daily_ohlc)["r_close"]
    sq = pd.DataFrame({
        "rs_neg": (r ** 2).where(r < 0, 0.0).where(r.notna()),
        "rs_pos": (r ** 2).where(r > 0, 0.0).where(r.notna()),
    })

    weekly = _to_monday_weekly(sq, how="sum")
    total = weekly["rs_pos"] + weekly["rs_neg"]
    weekly["rs_skew"] = ((weekly["rs_pos"] - weekly["rs_neg"]) / total).where(total > 0)
    return weekly


def realized_quarticity(daily_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Weekly realized quarticity and its HARQ scaling term.

    PRIOR-WEEK — pass through lag_for_join() before joining.

    RQ estimates the measurement error in the week's realized variance.
    HARQ (Bollerslev, Patton & Quaedvlieg 2016) uses sqrt(RQ) to scale the
    most recent lag's coefficient: when last week's vol estimate was noisy,
    lean on it less. Cheap to compute and it attacks the same weakness the
    HAR lags have — treating every past observation as equally reliable.

    Returns Monday-indexed:
        rq          (n/3) * sum of r^4, n = observations that week
        rq_sqrt     sqrt(rq), the HARQ interaction term
    """
    empty = pd.DataFrame(columns=["rq", "rq_sqrt"])
    if daily_ohlc.empty or "close" not in daily_ohlc.columns:
        return empty

    r = _daily_log_returns(daily_ohlc)["r_close"]
    quartic = pd.DataFrame({"_q": r ** 4, "_n": r.notna().astype(float)})

    weekly = _to_monday_weekly(quartic, how="sum")
    weekly["rq"] = (weekly["_n"] / 3.0) * weekly["_q"]
    weekly["rq_sqrt"] = np.sqrt(weekly["rq"].clip(lower=0.0))
    return weekly[["rq", "rq_sqrt"]]


# =============================================================================
# JOIN HELPERS
# =============================================================================

def lag_for_join(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag a PRIOR-WEEK candidate frame by one week for joining at week t.

    Thin, deliberately-named pass-through to feature_builder._lag_one_week so
    call sites read as an explicit statement about the lag boundary rather
    than an easily-missed underscore import.
    """
    return _lag_one_week(frame)


# Candidate columns grouped by lag class. gap_experiment builds its arms from
# these names, and any future promotion into feature_builder must preserve the
# grouping — a PRIOR_WEEK column joined without lag_for_join leaks.
SAME_WEEK_COLUMNS = ("weekend_gap", "abs_weekend_gap")
PRIOR_WEEK_COLUMNS = (
    "rv_overnight", "rv_intraday", "overnight_share",
    "rs_neg", "rs_pos", "rs_skew",
    "rq", "rq_sqrt",
)


def build_candidate_matrix(
    weekly_df: pd.DataFrame,
    daily_ohlc: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble every candidate on one Monday-indexed frame, each column
    already carrying the lag its class requires.

    The result is join-ready against a model_features frame:

        feats = feats.join(build_candidate_matrix(weekly, daily), how="left")

    Same-week columns land unlagged; prior-week columns are lagged one week.
    Mixing those up is the leak this function exists to prevent — callers
    never choose.
    """
    same_week = weekend_gap(weekly_df)

    prior_parts = [
        overnight_intraday_split(daily_ohlc),
        semivariance_split(daily_ohlc),
        realized_quarticity(daily_ohlc),
    ]
    prior_parts = [p for p in prior_parts if not p.empty]

    if prior_parts:
        prior = pd.concat(prior_parts, axis=1)
        prior = lag_for_join(prior)
    else:
        prior = pd.DataFrame(columns=list(PRIOR_WEEK_COLUMNS))

    out = same_week.join(prior, how="outer")
    out.index.name = "week_start"

    missing = [c for c in SAME_WEEK_COLUMNS + PRIOR_WEEK_COLUMNS
               if c not in out.columns]
    for col in missing:
        out[col] = np.nan
    if missing:
        log.warning(f"build_candidate_matrix: no data for {missing} — "
                    "columns emitted as all-NaN")

    return out
