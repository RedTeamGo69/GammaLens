# Gamma Lens

Streamlit dashboard for SPX-family options analysis.

- **Strike GEX** — dealer gamma exposure by strike, zero-gamma level, gamma walls, expected move
- **Spread Finder** — weekly credit-spread strike selection driven by a HAR volatility forecast

Tickers: SPX, XSP, SPY, QQQ, NDX, plus single names (NVDA, JPM, CAT, AMZN, AMD).

## Running locally

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

Requires `TRADIER_TOKEN`, `FRED_API_KEY`, and `DATABASE_URL` in `.streamlit/secrets.toml`.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Layout

| Path | Role |
|---|---|
| `phase1/` | GEX engine, market data, expected move |
| `range_finder/` | Weekly range model (HAR), features, DB |
| `ui_*.py` | Streamlit rendering |

A GitHub Actions cron captures snapshots at 9:45 ET on weekdays. See
[STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md) for deployment and [CLAUDE.md](CLAUDE.md) for
architecture notes and the constraints worth knowing before changing anything.
