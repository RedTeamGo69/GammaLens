"""
public_api — the ONLY Public.com touchpoint in Gamma Lens.

Hard data-source policy (do not relax): Public.com is used for exactly two
things — (a) reading the user's own fill/trade history, (b) preflighting
proposed orders for margin/cost estimates. ALL market data (chains, greeks,
OI, quotes, bars) comes from Tradier via phase1.data_client; nothing in this
package may ever fetch market data, and the rest of the app may import this
package only for those two purposes (today: ui_preflight's sync/preflight
buttons and scheduled_snapshot's daily sync). No order placement anywhere.
"""
