"""What the Spread Finder tab shows for one planning week: either why the
forecast is blocked, or the forecast → plan → tiers plus any regime-shift
warning. Pure (no Streamlit); the tab renders it.

This is the view-model the tab's render function used to assemble inline,
where none of it was reachable by a test.
"""
from dataclasses import dataclass

from range_finder.feature_builder import ForecastRowBlocked, select_forecast_row
from range_finder.recommendations import build_recommendations
from range_finder.spread_finder_rules import RegimeShift, detect_regime_shift


@dataclass(frozen=True)
class SpreadFinderView:
    week_start: str
    blocked: "ForecastRowBlocked | None" = None
    regime_shift: "RegimeShift | None" = None
    feature_row: object = None
    forecast: "dict | None" = None
    plan: object = None
    tiers: "list | None" = None

    @property
    def ready(self) -> bool:
        return self.blocked is None


def build_spread_finder_view(*, features, week_start, result, feature_cols,
                             reference, vix, live_vix, ticker,
                             side_share_q=None, chain_quotes=None,
                             weekly_em=None, conn=None,
                             model_name=None) -> SpreadFinderView:
    """The planning week's view, or a blocked view naming why.

    ``live_vix`` must be the genuinely fetched value (None when the fetch
    failed), never the input's display default: it drives the breaker.
    """
    feature_row, blocked = select_forecast_row(features, week_start)
    if blocked is not None:
        return SpreadFinderView(week_start, blocked=blocked)
    forecast, plan, tiers = build_recommendations(
        result=result, feature_row=feature_row, feature_cols=feature_cols,
        reference=reference, vix=vix, week_start=week_start, ticker=ticker,
        side_share_q=side_share_q, chain_quotes=chain_quotes,
        weekly_em=weekly_em, conn=conn, model_name=model_name,
    )
    return SpreadFinderView(
        week_start,
        regime_shift=detect_regime_shift(live_vix, feature_row.get("vix_close")),
        feature_row=feature_row, forecast=forecast, plan=plan, tiers=tiers,
    )


def _label(ts) -> str:
    return ts.strftime("%Y-%m-%d") if ts is not None else "missing"


def blocked_message(view: SpreadFinderView) -> str:
    """The tab's markdown explanation for a blocked view."""
    b = view.blocked
    if b.reason == "missing":
        newest = _label(b.newest_week) if b.newest_week is not None else "none"
        return (
            f"⚠️ **Forecast blocked:** no feature row exists for {view.week_start}. "
            f"The newest persisted row is {newest}. Run **Weekly "
            "Setup** (or **Refresh Data** + **Rebuild Features**) before "
            "using this week's strikes."
        )
    return (
        f"⚠️ **Forecast blocked:** the {view.week_start} row uses weekly path "
        f"inputs from **{_label(b.source_week)}**, but it requires "
        f"**{_label(b.required_week)}**. A provisional next-week row "
        "created before Friday's close cannot be reused after the week "
        "rolls. Run **Weekly Setup** (or **Refresh Data** + **Rebuild "
        "Features**) to refresh the HAR inputs. The saved Monday-open "
        "strike anchor will remain unchanged."
    )


def regime_shift_message(shift: RegimeShift) -> str:
    """The tab's markdown warning for a VIX regime shift."""
    return (
        f"⚠️ **VIX regime shift detected ({shift.severity})** — "
        f"live VIX **{shift.live_vix:.1f}** vs trailing "
        f"feature VIX **{shift.trailing_vix:.1f}** "
        f"(**{shift.ratio:.2f}×**).\n\n"
        "The HAR model's features lag by one week, so the forecast below "
        "is anchored to the pre-spike vol regime. Short strikes sized "
        "against this forecast are likely **too narrow**. Check the live "
        "weekly expected-move band on the strike map (it reflects the "
        "current straddle) and treat any short strike inside it as too "
        "risky — or, better, skip the trade until features catch up."
    )
