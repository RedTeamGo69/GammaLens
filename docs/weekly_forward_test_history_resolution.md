# Historical source resolution — September 8, 2026 ET

The historical-data blocker is resolved by `tradier-reviewed-whole-bars-v1`.
The pre-merge live check completed at 2026-09-09T01:32Z with no capability
blockers. This document records source review; release/deployment/activation
must still complete the separately authorized sequence. No production forecasts
were created by any check.

## Original defects and current evidence

The [original audit](weekly_forward_test_history_review.md) preserves the raw
SPX missing-OHL response and all eight SPY closes below their lows. Range and
individual-date requests agreed before normalization. The refreshed September 8
Tradier responses now contain valid SPX March 19 OHL and the corrected August 31
open, plus several corrected recent stock opens. These upstream changes are
evidence against a local date/normalization cause. They do not erase the original
responses. A recurrence of the missing SPX fields fails admission.

SPY still has 24 pre-dividend close discrepancies, including the original eight
invalid bars. Their amounts agree with independently retrieved cash-dividend
metadata, while the OHL are not coherently dividend-adjusted. This identifies a
source adjustment inconsistency; the vendor's exact internal calculation remains
an inference. [Tradier documents exchange-dependent dividend adjustments](https://docs.tradier.com/docs/historical-data)
and provides no supported history adjustment switch. Nothing here adds a
dividend amount to a close or manufactures an OHLC field.

AAPL/AMD September 11, 2023 and AAPL September 12 also retain source defects.
The September 11 AAPL primary bar is 179.45/179.55/179.45/179.55;
the accepted whole Yahoo bar is
180.07000732421875/180.3000030517578/177.33999633789062/179.36000061035156.
A fresh Massive response independently gives 180.07/180.30/177.34/179.36.
AMD's primary September 11 bar is 104.8 in all four fields; its accepted whole
bar is approximately 107.32/107.51/103.00/105.32, corroborated by Robinhood and
the prior Massive pull. Corresponding weekly differences are reviewed too.

The complete machine-readable [review catalog](../range_finder/forward_test/history_resolutions.json)
contains all 52 current disputed daily/weekly bars, exact original and alternative
OHLC, accepted source, justification, raw witness rows, request parameters,
retrieval times and response hashes. Twenty decisions retain Tradier against a
conflicting Yahoo value; 32 replace an entire source bar. This is evidence
selection before fitting, never selection based on favorable predictions.

## Source policy and independent witnesses

Tradier remains primary. A single bounded Yahoo daily chart request per ticker
provides raw `indicators.quote` OHLC on the current split basis, without dividend
adjustment, `auto_adjust`, repair, pre/post-market inclusion or `adjclose` use.
SPX is explicitly `^GSPC`, USD, INDEX, America/New_York; ETFs and stocks have
their own checked identities. Timestamps become New York session dates. Yahoo
Monday weekly OHLC is aggregated only after exact exchange-session coverage and
full daily OHLC validity pass. No missing, duplicate, future or partial-final-week
row is dropped to make a series pass.

All consumed daily closes and every weekly OHLC field are compared. Daily OHL
must remain valid even though they do not enter features. Four current SPY daily
OHL-only discrepancies are archived as warnings where consumed closes agree and
weekly OHLC is independently checked. Observations used for breach/path scoring
continue to require the original strict full OHLC/minute rules; this policy is
never used to repair an observation.

Comparison precision is explicitly 0.0051 USD, covering half-cent reporting and
Yahoo float32 representation. It is a methodology-versioned source comparison,
not a widening of price ranges, strike boundaries or admission rules. Input
prices retain their exact source values. An exception must match the recorded
ticker, cadence, date and both entire OHLC signatures. Replacement is the entire
Yahoo source bar, including its volume, not a field-by-field mixture. A changed
signature, unknown disagreement, absent corroborating source, incompatible
instrument/basis, incomplete witness or missing session returns
`UNAVAILABLE_HISTORY`. Both source responses and the selected independent
evidence are archived with the immutable forecast input snapshot.

Robinhood's accessible split-only, regular-session stock history supplies
independent whole-OHLC evidence for all 24 SPY corrections, including the sixth
year. Its SPX index history required bounded two-year requests. Its UTC-midnight
daily labels are calendar labels, not timestamps to shift to the preceding New
York date. Interpolated bars were excluded as witnesses; one real session
(2019-02-21) was interpolated, and none of the selected witness weeks uses it.
Robinhood is not adopted as a blanket substitute: its AAPL September 11 open
conflicts with Yahoo/Massive. That correction uses Massive instead.

FRED SP500, sourced from S&P Dow Jones Indices, resolves the official SPX
November 3, 2020 close at 3369.16 against the 3369.02 primary/Robinhood close.
Robinhood corroborates the candidate OHL; FRED corroborates its close. The
accepted input is still one coherent Yahoo bar, not synthesized fields.

Current Massive probes reconfirm stock access for September 2023 but
`NOT_ENTITLED` for SPX index bars and SPY December 2020. Stock access was not
assumed to cover indices or the full training horizon. Public's connected price
history corroborates recent SPX data, but ten-year daily aggregation is unsupported
and its long weekly response uses Wednesday labels. It was not used for Monday
training weeks. The separate local Public adapter reports an invalid secret;
no credentials were extracted from connector sessions. No paid access was added.
These connectors supply preserved, reviewed historical witnesses. Scheduled
capture uses existing Tradier credentials and the bounded Yahoo request only.

## Model dependencies and interactive behavior

The six-year fit cutoff, 30-day daily warm-up, COVID exclusion, feature formulas,
model specifications, GEX exclusion, opening anchors, tiers, side-placement
quantile and all deadlines remain unchanged. Weekly OHLC supplies range/log-range
targets, weekly returns, HAR lags, the path return features and the pooled
side-share quantile. Daily close supplies 5/20-session historical volatility and
its ratio in extended/full models. M1/M2 do not consume daily closes directly,
but weekly corrections can affect every model.

The existing legacy database determines the older side-placement horizon. Its
prices and derived columns are no longer silently prepended: a fresh validated
weekly response covers the same full horizon. That also resolves the old AAPL
dividend-adjusted basis and a discrepant old SPY week. No legacy row, saved model,
research result or calibration is overwritten. The new policy and catalog hashes
are part of the stable methodology. Individual weekly fits, training data,
coefficients, covariance and source receipts retain separate identities.

The interactive Spread Finder still reads its existing stored features/history
and fallback path. Its legacy data can be stale or contain the audited values;
therefore its live displayed recommendations may differ from the automated
study's newly accepted inputs. The Forward Test page states this difference.
Regression tests feed the actual accepted histories into the shared feature
builder, apply the same six-year cutoff, and compare UI/headless recommendations
for all four models, four tickers and four tiers. There is no formula fork.

## Live pre-release coverage

Checked against the intended accounts and read-only production database at
2026-09-09T01:32:08Z. The market was closed. Target registration is the week
2026-09-14; September 8 admission has passed. The latest fully completed week
available for this midweek review ends September 4. Capture still requires
September 11 history when that session has actually completed.

| Ticker | Accepted daily sessions | Accepted weekly bars | Whole-bar replacements | Exact calendar / price gate |
|---|---:|---:|---:|---|
| SPX | 1,521 | 535 from 2016-06-06 | 1 | pass |
| SPY | 1,521 | 534 from 2016-06-13 | 25 | pass |
| AAPL | 1,521 | 326 from 2020-06-08 | 3 | pass |
| AMD | 1,521 | 534 from 2016-06-13 | 3 | pass |

The rolling daily requirement starts August 15, 2020 (first exchange session
August 17), through September 4, 2026. This date movement follows the prospective
week moving one week forward; no history was shortened to avoid a defect.
Daily/weekly final closes agree after reviewed selection. Every future capture
repeats the same gate; a new discrepancy requires review and cannot silently pass.

Quotes have correct symbols, opens and last values, dated September 8. Current
September 18 expiry chains contain verified call/put counts of SPXW 452/452,
SPY 321/321, AAPL 104/104 and AMD 154/154, all standard 100-unit contracts.
September 4 09:45–16:00 minute probes return 375 bars per ticker. Cboe VIX,
VIX9D and VIX3M include September 8 and the required reviewed prior session.
FRED DGS10/DGS2/DFF include September 4; event lists cover the target week
through December 2026. The intended Neon fingerprint is `36d4cccd11918975`;
the read-only review found zero `ft_*` tables before release.

Opening-session freshness, actual live feed delay, the September 11 completed
history, capture completion before the database-clock deadline and post-close
first-session minute completeness remain session-dependent. No quote delay is
inferred from subscription names or out-of-session timestamp ages. Activation
waiting for a future session is not evidence that its first capture succeeded.

Full sanitized source bodies and manifest are in the local
`forward-history-live-20260909` and `forward-history-resume-20260909` evidence
directories. The compressed isolated regression fixture retains the full current
windows and original response receipts. `build_forward_history_resolutions.py`
reproduces the catalog from explicit reviewed decisions and independent raw
evidence, asserting complete witness weeks and rejecting changed evidence hashes.

## Verification before release

The exact project command `.venv\Scripts\python.exe -m pytest -q
--junitxml=<local-evidence>/release-tests.xml` completed successfully: **590
passed in 102.00 seconds**, zero failures/skips. This includes 14 unrelated
untracked candidate-feature tests already present locally; the committed GitHub
suite therefore contains 576 tests. The two PostgreSQL tests ran against the
separate disposable loopback PostgreSQL 17.11 cluster on port 55441. They verify
schema/migration idempotency, constraints, immutable triggers, concurrent
single-winner capture and rollback on both database-clock deadline checks. The
older stopped `gammalens-forward-postgres` directory was not touched.

The production-fit two-week regression reports one SPY/M2/Point methodology
group with `close_n=2` and `path_n=2`, despite different coefficients, covariance,
fit hashes, source-input hashes and training cutoffs. Removing the actual VIX
feature for the third fixture week creates a separate group with `close_n=1`
and `path_n=1`. These are isolated test results, never production forecasts.

The repository variable remains false during release. Streamlit management is
accessible. The cron-job.org console still requires sign-in; no additional
dispatcher or GEX modification was made. The separate GitHub workflow retains
09:45/09:55/10:05 ET capture attempts, the exclusive 10:15 deadline, 18:10 ET
observations and `cancel-in-progress: false`. Its punctuality is not guaranteed.
