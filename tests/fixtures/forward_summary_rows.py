"""Synthetic examples for weekly recap tests; never written to the study."""
from range_finder.forward_test.config import MODELS
from range_finder.recommendations import TIER_KEYS, TIER_LABELS


def summary_rows():
    rows = []
    for ticker in ("AAPL", "AMD", "SPX", "SPY"):
        for model in MODELS:
            for tier, label in zip(TIER_KEYS, TIER_LABELS):
                reference, close, put, call = 100.0, 101.0, 90.0, 110.0
                breach = tier == "lower_pi" and (ticker == "SPX" or
                            (ticker == "SPY" and model in MODELS[:2]))
                if ticker == "AMD":
                    reference, close, put = 486.28, 559.82, 410.0
                    call = {"lower_pi": 515, "point": 530, "pi_upper": 565, "effective": 570}[tier]
                    if model == "M1_baseline":
                        call = {"pi_upper": 557.5, "effective": 560}.get(tier, call)
                    elif model == "M2_vix":
                        call = {"pi_upper": 550, "effective": 555}.get(tier, call)
                    breach = close > call
                inside = put <= close <= call
                rows.append(dict(study_id="SYNTHETIC", cohort="deterministic_fixture",
                    week_start="2026-09-14", ticker=ticker, model=model,
                    model_version=f"fixture-{ticker}-{model}", tier=tier, tier_label=label,
                    forecast_id=f"{ticker}-{model}-{tier}", reference=reference,
                    final_close=close, put_short=put, call_short=call,
                    range_width_ratio=(call-put)/reference, close_eligible=True,
                    path_eligible=True, close_inside=inside, either_breach=breach,
                    returned_inside=inside and breach, capture_status="captured", status="final",
                    classification="Synthetic outcome", settlement_status="unverified",
                    expiration="2026-09-18", scored_at="2026-09-18T22:10:00+00:00"))
    return rows
