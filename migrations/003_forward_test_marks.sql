-- Explicit, additive migration. Never called by Streamlit/init_all_tables.
-- Executable marks for every frozen spread: one row per forecast contract per
-- sample, so stops and profit targets price on real two-sided quotes instead of
-- hold-to-close only. Append-only (the PostgreSQL trigger lives in 002).
CREATE TABLE IF NOT EXISTS ft_marks (
    forecast_id TEXT NOT NULL REFERENCES ft_forecasts(forecast_id),
    leg TEXT NOT NULL, symbol TEXT NOT NULL,
    bid DOUBLE PRECISION, ask DOUBLE PRECISION, bid_at TEXT, ask_at TEXT,
    underlying_last DOUBLE PRECISION, underlying_at TEXT, collected_at TEXT NOT NULL,
    PRIMARY KEY(forecast_id, symbol, collected_at)
);
