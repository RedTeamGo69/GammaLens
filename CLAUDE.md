# Gamma Lens

Streamlit dashboard for SPX-family options analysis. Two tabs:

- **Strike GEX** — dealer gamma exposure by strike, zero-gamma level, walls, expected move
- **Spread Finder** — weekly credit-spread strike selection from a HAR volatility forecast

Python 3.11, Streamlit, Neon Postgres. Deployed on Streamlit Cloud; a GitHub Actions cron
captures daily snapshots.

## Commands

```bash
.venv/Scripts/python.exe -m streamlit run streamlit_app.py    # run the app
.venv/Scripts/python.exe -m pytest -q                          # 402 tests, ~20s
.venv/Scripts/python.exe -m pytest range_finder/tests -q        # model tests only
```

`gh` is installed but sessions started before 2026-08-07 have a stale PATH — call
`"C:\Program Files\GitHub CLI\gh.exe"` by full path.

## Layout

Three layers, roughly in dependency order:

| Path | Role |
|---|---|
| `phase1/` | GEX engine + market data. `gex_engine.py`, `zero_gamma.py`, `expected_move.py`, `key_levels.py`, `ticker_config.py`, `data_client.py` |
| `range_finder/` | Weekly range model. `har_model.py` (the forecast), `feature_builder.py`, `spread_levels.py`, `db.py`, `event_calendars.py`, `calibration.py` |
| `ui_*.py` (root) | Streamlit rendering. `ui_charts.py`, `ui_spread_finder.py`, `ui_theme.py`, `ui_controls.py`, `ui_sidebar.py`, `ui_tv_export.py` |

Entry points: `streamlit_app.py` (`main()` at line 379), `scheduled_snapshot.py` (cron),
`bootstrap_range_finder.py` (one-time DB seed).

Tabs are declared in `ui_theme.py::TABS`. Adding one means touching that list *and* the
dispatch in `streamlit_app.py`.

`range_finder/*_experiment.py` files are offline research scripts, not app code — they exist
to re-run evidence for the decisions below and are never imported by the app.

## Data sources

- **Bars/quotes:** Tradier (`TRADIER_TOKEN`) — primary
- **VIX family:** Cboe CDN (`cboe_data.py`)
- **Macro:** FRED (`FRED_API_KEY`)
- **yfinance:** fallback only. Do not add primary yfinance calls.

## Database

Neon Postgres via `DATABASE_URL`. `range_finder/db.py::init_all_tables` is idempotent and
runs on cron startup.

Live tables: `weekly_spx`, `weekly_underlying`, `daily_spx`, `model_features`,
`daily_model_features`, `saved_models`, `event_flags`, `earnings_flags`, `gex_snapshots`,
`gex_inputs`, `em_snapshots`, `macro_daily`, `spread_log`, `weekly_setup`,
`interval_calibration`, `forecast_log_daily`, `spread_log_daily`, `event_flags_daily`.

The `*_daily` tables and `daily_spx` are orphaned history from the 0DTE removal — nothing
reads or writes them, and `init_all_tables` no longer creates them.

**Dropping a table is not enough while the Streamlit Cloud container is on old code.**
`init_all_tables` uses `CREATE TABLE IF NOT EXISTS`, so a stale container recreates the
shells (empty — no data returns) on every new session. Proven on 2026-08-07: the five
Cockpit/Pre-Flight tables were dropped and verified gone, then reappeared with fresh OIDs
the moment the deployed app was opened, while HEAD contained zero references to them.

**Streamlit Cloud did not auto-redeploy on push** — the container kept serving pre-removal
code for hours after the commit landed, so the live app still showed the two deleted tabs.
A "session reset" restarts the script but not the container, which is why the old code stayed
resident. Only **⋮ → Reboot app** on share.streamlit.io picks up a new commit.

So the order matters: reboot first, confirm the UI reflects the new code, *then* drop tables
and re-verify. Dropping before the reboot just gets undone. OID ordering (`pg_class.oid`,
highest = newest) is the cheap way to tell a recreated table from an original.

## Cron

`.github/workflows/scheduled_snapshot.yml` — 9:45 ET on weekdays (two cron lines to cover
EDT/EST). Matrix: `SPX, XSP, SPY, QQQ, NDX, NVDA, JPM, CAT`. Index/ETF capture daily;
single names run weekly-only on the week's first trading day.

`TICKER_CONFIG` (`phase1/ticker_config.py`) has explicit entries for SPX, XSP, QQQ, SPY,
NDX, AMZN, AMD; the single names in the matrix resolve through the dynamic-config path.

## Hard-won constraints

These are load-bearing. Each one cost a debugging session or a walk-forward study.

- **`st.html` strips inline SVG.** Streamlit's sanitizer drops `<svg>`. All charts are
  built from `div` + CSS. Never reach for inline SVG in a rendered chart.
- **Spread Finder strikes lock to Monday's open** Mon–Thu, with a self-heal capture. Strikes
  drifting mid-week is a bug, not a feature.
- **`TRAIN_WINDOW_YEARS = 6`** (`har_model.py:52`). 6y beat 10y in walk-forward testing. The
  DB deliberately holds 10y of rows; production reads pin `min_date=train_window_min_date()`.
- **COVID rows are excluded** (`exclude_covid=True`, `feature_builder.py:371`;
  `COVID_START = 2020-03-01`). Tested 2026-07-10: including them lifts index OOS R² but
  over-widens prediction intervals. Failed the gate 62/80. `covid_experiment.py` re-runs it.
- **Conformal prediction is OFF** (`CONFORMAL_ENABLED = False`, `conformal.py:41`) — by
  verdict, not by omission.
- **Model ranges are not persisted.** They recompute live. Reconstructing a past week means
  doing it before the Monday cron overwrites `saved_models`.
- **Nav state is not URL-persisted.** Ticker, tab, and recents live only in `session_state`,
  so a Streamlit Cloud session reset wipes all three at once — one bug, not three.
- **xlsx export scores by ticker** via an `INDIRECT` sanitizer. `IFERROR(SUMIF(INDIRECT(...)))`
  silently returns 0 in Excel, so the sanitizer is what makes the scoreboard correct.

## Deliberately removed — do not re-add without new evidence

| Removed | When | Why |
|---|---|---|
| Monday Cockpit, Pre-Flight, `public_api/` | 2026-08-07 | Not useful. Both tabs, the whole Public.com fills pipeline, and all 5 of their DB tables dropped. |
| 0DTE finder + daily pipeline | 2026-07-02 | 1,036-session audit: the VRP verdict didn't discriminate. |
| VRP green/red verdict | 2026-08-03 | Straddle-implied = 0.703× VIX-implied, so VRP 1.10 sat at the median with no edge. |
| XND | — | Too illiquid; Tradier won't quote it. |
| charm | — | Low signal. |

`event_calendars.py` kept `PPI_DATES`/`PCE_DATES` after the Cockpit removal for one reason:
`calendar_staleness_warnings` reads all five lists to drive the Spread Finder banner. Its only
other live entry point is `build_event_flags` (tier-1 FOMC/CPI/NFP/OpEx → `event_flags`).

## Conventions

- Match surrounding style; this codebase favors dense explanatory comments on *why*, not what.
- Many small focused modules over large ones.
- Tests live in `tests/`, `phase1/tests/`, `range_finder/tests/`. Add coverage with behavior.
- Pine Script: consult the `pine-docs` MCP before writing any — verify signatures, and follow
  the GL1 indicator rules (anchor plot, `islast`-only, one statement per line).
- Conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`).
