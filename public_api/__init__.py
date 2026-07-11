"""
public_api — the ONLY Public.com touchpoint in Gamma Lens.

Hard data-source policy (do not relax): Public.com is used for exactly two
things — (a) reading the user's own fill/trade history, (b) preflighting
proposed orders for margin/cost estimates. ALL market data (chains, greeks,
OI, quotes, bars) comes from Tradier via phase1.data_client; nothing in this
package may fetch market data, and nothing outside this package may import
the Public client. No order placement anywhere.
"""
