# SPX GEX Dashboard — Streamlit Deployment Guide

## Deploy to Streamlit Cloud (Free — Access from Anywhere)

### 1. Push to GitHub

Create a repo with this structure:

```
your-repo/
├── streamlit_app.py
├── requirements.txt
├── phase1/
│   ├── __init__.py
│   ├── config.py
│   ├── gex_engine.py
│   ├── model_inputs.py
│   ├── market_clock.py
│   ├── data_client.py
│   ├── parity.py
│   ├── quote_filters.py
│   ├── rates.py
│   ├── liquidity.py
│   ├── confidence.py
│   ├── staleness.py
│   ├── wall_credibility.py
│   ├── scenarios.py
│   ├── expected_move.py
│   └── run_metadata.py
```

**Important:** Do NOT commit your API keys. Use Streamlit secrets instead.

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo, branch, and `streamlit_app.py`
5. Click **Advanced settings** → paste your secrets:

```toml
PUBLIC_SECRET = "your_public_api_secret"
FRED_API_KEY = "your_fred_api_key"
```

> Generate the Public secret at **public.com → Settings → Security → API**.
> It is long-lived but revocable; the app exchanges it for short-lived
> access tokens automatically. Optional secrets: `PUBLIC_ACCOUNT_ID` (pin a
> specific account if your secret has more than one) and
> `PUBLIC_TOKEN_VALIDITY_MIN` (access-token lifetime, default 120).

6. Click **Deploy**

Your app will be live at `https://your-app-name.streamlit.app` — accessible from any device.

### 3. Make it Private (Optional)

Streamlit Cloud apps are public by default on the free tier. Options:

- **Viewer auth:** Streamlit Cloud supports Google OAuth for viewer gating (paid teams plan)
- **Self-host:** Deploy on a $5/mo VPS (DigitalOcean, Railway, Fly.io) behind basic auth
- **Render.com:** Free tier with `streamlit run` as the start command

## Features

- **Auto-refresh:** Toggle in sidebar for 90-second refresh cycles
- **Mobile-friendly:** Streamlit responsive layout works on phones
- **Expected Move panel:** ATM straddle, overnight move, session classification
- **EM levels on charts:** Purple dotted lines on both Strike GEX and Profile charts
- **All existing features:** Zero gamma sweep, wall credibility, scenarios, heatmaps
