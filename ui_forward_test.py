"""Read-only Forward Test view. No quotes, model refits, or schema changes."""
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st

from range_finder.forward_test.results import load_results, metrics, scoreboard, build_workbook
from range_finder.forward_test.config import UNIVERSE, MODELS, COHORT, DEFAULT_STUDY_ID


@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def load_snapshot():
    from range_finder.forward_test.store import Store
    url = os.environ.get("FORWARD_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        url = st.secrets.get("DATABASE_URL", "")
    store = Store.postgres(url)
    try:
        studies = store.query("SELECT study_id, start_week FROM ft_studies ORDER BY created_at DESC")
        records = [r for s in studies for r in load_results(store, s["study_id"])]
        runs = store.query("SELECT * FROM ft_runs ORDER BY started_at DESC LIMIT 30")
        last_ok = store.query("SELECT * FROM ft_runs WHERE status='ok' AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1")
        if last_ok and all(r['run_id'] != last_ok[0]['run_id'] for r in runs):
            runs.extend(last_ok)
        return records, runs, studies
    finally:
        store.close()


@st.cache_data(show_spinner=False)
def deployed_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent, text=True, timeout=5,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return 'Unavailable'


def render_forward_test(rows=None, runs=None, studies=None):
    st.title("Forward Test")
    st.caption("Frozen weekly predictions for SPX, SPY, AAPL, AMD, META, TSLA and HOOD. "
               "Observational results; no trades are submitted.")
    if rows is None:
        try:
            rows, runs, studies = load_snapshot()
        except Exception as exc:
            st.info("Forward Test is not available yet. Apply the additive migrations and register the study using the activation instructions.")
            st.caption(f"Database check: {type(exc).__name__}. If already activated, check the database connection and the Weekly Spread Finder Forward Test workflow.")
            return
    runs = runs or []
    studies = studies or []
    # Select the registered study before checking for empty results. Otherwise
    # a fresh restart with no forecasts falls back to the retired study's rows.
    study_ids = list(dict.fromkeys([s['study_id'] for s in studies] + [r['study_id'] for r in rows]))
    if study_ids:
        preferred = DEFAULT_STUDY_ID if DEFAULT_STUDY_ID in study_ids else study_ids[0]
        # Superseded registrations with no slots/results are not result archives.
        populated = {r['study_id'] for r in rows}
        study_ids = [sid for sid in study_ids if sid == preferred or sid in populated]
        study_ids = [preferred] + [sid for sid in study_ids if sid != preferred]
        if len(study_ids) > 1:
            from ui_forward_summary import _choose
            dates = {s['study_id']: s['start_week'] for s in studies}
            def label(sid):
                status = 'Current study' if sid == preferred else 'Archived study'
                return f"{status} · {dates.get(sid, sid)}"
            selected = _choose('Study', study_ids, 'ft_active_study', format_func=label)
        else:
            selected = preferred
        rows = [r for r in rows if r['study_id'] == selected]
        studies = [s for s in studies if s['study_id'] == selected]
        runs = [r for r in runs if r['study_id'] == selected]
        if selected == DEFAULT_STUDY_ID:
            st.caption("Tracking starts at the first trading session's 9:30 AM ET open. "
                       "Opening prices anchor the ranges; actual forecast save times are recorded separately.")
    st.session_state.setdefault("ft_view", st.session_state.get("_ft_view_last", "Summary"))
    with st.container(horizontal=True, vertical_alignment="center"):
        view = st.segmented_control("Results view", ["Summary", "Detailed data"],
                                    key="ft_view", on_change=_keep_view,
                                    label_visibility="collapsed")
        if st.button("Refresh results"):
            load_snapshot.clear()
            st.rerun()
    st.session_state["_ft_view_last"] = view
    if not rows:
        st.info("No frozen predictions yet. Registered studies are waiting for an eligible opening-week capture.")
    if view == "Detailed data":
        filtered = _render_details(rows, studies)
    else:
        from ui_forward_summary import render_summary
        filtered = render_summary(rows, studies)
    st.download_button("Download Excel results", build_workbook(filtered),
                       file_name="Gamma_Lens_Forward_Test.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if runs and runs[0]["status"] != "ok":
        st.warning("The latest collection run needs attention. See Capture and scoring health below.")
    with st.expander("Capture and scoring health"):
        _render_health(rows, runs)


def _keep_view():
    if st.session_state.get("ft_view") is None:
        st.session_state["ft_view"] = st.session_state.get("_ft_view_last", "Summary")


def _render_details(rows, studies):
    st.caption("New predictions use Tradier price history. Earlier predictions retain their original inputs. "
               "Spread Finder uses its existing history, so recommendations can differ from this study.")
    for study in studies:
        st.caption(f"Registered study: {study['study_id']} · First eligible week: {study['start_week']}")
    filtered = list(rows)
    keys = [("study_id", "Study"), ("cohort", "Capture cohort"), ("ticker", "Ticker"),
            ("model", "Model"), ("tier_label", "Tier"), ("model_version", "Model version"), ("week_start", "Week")]
    for i in range(0, len(keys), 3):
        for column, (key, label) in zip(st.columns(min(3, len(keys) - i)), keys[i:i + 3]):
            with column:
                options = {str(r[key]) for r in rows if r.get(key) is not None}
                # Registration is visible before the first capture without
                # inventing pending forecasts or seeding synthetic slots.
                defaults = {'study_id': [s['study_id'] for s in studies],
                            'week_start': [s['start_week'] for s in studies],
                            'ticker': UNIVERSE, 'model': MODELS, 'cohort': [COHORT],
                            'tier_label': ['Lower PI', 'Point Estimate', '80% PI Upper', 'Effective (+buffer)']}
                options = sorted(options.union(defaults.get(key, [])))
                widget_key = f"ft_filter_{key}"
                saved_key = f"_ft_filter_{key}"
                previous = st.session_state.get(widget_key, st.session_state.get(saved_key, []))
                st.session_state[widget_key] = [value for value in previous if value in options]
                selected = st.multiselect(label, options, key=widget_key)
                st.session_state[saved_key] = list(selected)
                if selected:
                    filtered = [r for r in filtered if str(r.get(key)) in selected]
    summary = metrics(filtered)
    def pct(value):
        return "—" if value is None else f"{value:.1%}"
    columns = st.columns(4)
    for col, label, value, count in zip(columns,
            ("Weekly close hit rate", "Breach rate", "Breach recovery rate", "Average range / reference"),
            (summary["close_hit_rate"], summary["breach_rate"], summary["recovery_rate"], summary["average_width_ratio"]),
            (summary["close_n"], summary["path_n"], summary["recovery_n"], summary["width_n"])):
        with col:
            st.metric(label, pct(value))
            st.caption(f"Eligible tier records: {count}")
    st.caption(f"Put-side close failures: {summary['put_failures']} · Call-side close failures: {summary['call_failures']} · "
               f"Pending: {summary['pending']} · Missed: {summary['missed']} · "
               f"Unavailable: {summary['unavailable']} · Incomplete/invalid: {summary['incomplete']}")
    if len({r.get("model_version") for r in filtered if r.get("forecast_id")}) > 1:
        st.caption("Overview includes multiple model versions. The comparison table keeps versions and cohorts separate.")
    st.caption("Close statistics require a valid final close. Breach statistics require complete prospective coverage. "
               "Recovery rate counts inside closes among records with a verified breach and valid close/path data.")
    st.subheader("Model comparison")
    if filtered:
        comparison = pd.DataFrame(scoreboard(filtered))
        comparison['tier'] = comparison['tier'].map(dict(zip(
            ('lower_pi', 'point', 'pi_upper', 'effective'),
            ('Lower PI', 'Point Estimate', '80% PI Upper', 'Effective (+buffer)'))))
        comparison_fields = ['ticker', 'model', 'tier', 'close_hit_rate', 'close_n', 'breach_rate', 'path_n',
                             'recovery_rate', 'recovery_n', 'average_width_ratio', 'put_failures', 'call_failures',
                             'pending', 'missed', 'unavailable', 'incomplete', 'cohort', 'model_version', 'study_id']
        st.dataframe(comparison.reindex(columns=comparison_fields), hide_index=True)
        fields = ["week_start", "ticker", "model", "tier_label", "put_short", "call_short", "final_close",
                  "classification", "put_breach", "call_breach", "earlier_close_outside", "returned_inside",
                  "close_on_boundary", "available_at", "expiration", "settlement_status", "error"]
        st.subheader("Predictions and results")
        st.dataframe(pd.DataFrame(filtered).reindex(columns=fields), hide_index=True)
    return filtered


def _render_health(rows, runs):
    st.caption(f"App revision: {deployed_revision()}")
    good = next((r for r in runs if r["status"] == "ok" and r.get("finished_at")), None)
    st.caption(f"Latest successful run: {good['finished_at'] if good else 'No successful run recorded'}")
    last_capture = max((r['available_at'] for r in rows if r.get('available_at')), default='None recorded')
    last_score = max((r['scored_at'] for r in rows if r.get('scored_at')), default='None recorded')
    st.caption(f"Latest frozen forecast: {last_capture} · Latest saved score: {last_score}")
    if runs:
        latest = runs[0]
        st.write(f"Latest run: {latest['status']} · {latest['started_at']}")
        if latest["status"] == "running":
            st.warning("The latest run has no completion record. Check the workflow for interruption or timeout.")
        details = json.loads(latest["payload_json"])
        for error in details.get("errors", [])[:12]:
            st.warning(error)
        with st.expander("Recent runs"):
            st.dataframe(pd.DataFrame(runs).drop(columns=["payload_json"], errors="ignore"), hide_index=True)
    st.caption("SPX uses SPXW PM-settled contracts. Stock/ETF close results are theoretical expiration proxies. "
               "Official settlement, when verified, is stored separately. A temporary breach is not a weekly-close failure.")
