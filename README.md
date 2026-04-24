# IMC Prosperity — Round 3 Trader

Single-file trader (`trader.py`) for the Round 3 options market. Submit the
file as-is to Prosperity; `datamodel.py` in this repo is only used locally.

## Files

- `trader.py` — the submission. Contains `Trader`, all strategies, option
  math, and an embedded backtester.
- `datamodel.py` — local shim of Prosperity's datamodel (no `jsonpickle`
  dependency, unlike the one under `backtester/`). Prosperity's runtime
  shadows this at deploy time.

## Running the backtest

```
mkdir -p data
# copy prices_round_3_day_0.csv, prices_round_3_day_1.csv, prices_round_3_day_2.csv
# and trades_round_3_day_0.csv, ..._1.csv, ..._2.csv into ./data/
python3 trader.py
```

If `./data/` is missing or empty, the script exits with a clear message — no
error. Change `PARAMS["backtest"]["data_dir"]` if your CSVs live elsewhere.

Fill models (set `PARAMS["backtest"]["fill_model"]`):

- `hybrid_scaled` (default) — cross-book fills, plus passive fills sized from
  historical trades. A quote strictly better than a trade price gets 100% of
  the trade quantity; equal-price quotes get `passive_fill_ratio` (default
  0.5). This rewards genuinely competitive quoting.
- `hybrid_flat` — passive fills always at `passive_fill_ratio`, regardless of
  price advantage.
- `cross_only` — pessimistic, no passive fills. Use as a sanity check.

Trades are matched in the window `[ts, ts + trade_window)` — `prices_*.csv`
has one snapshot every 100 timestamps, but `trades_*.csv` is sparse, so
exact-timestamp matching would miss most fills.

## The 5400 edge and how the trader exploits it

Historical residuals (market IV minus smile-fit IV) for `VEV_5400`:

| day | residual |
|-----|----------|
| 0   | −1.64%   |
| 1   | −1.32%   |
| 2   | −0.88%   |

Linear projection to `t_global = 3.0` (live round start) ≈ −0.77%, and the
edge plausibly closes by `t_global = 5.0` (mid-round). The trader encodes
this via a **time-varying long inventory target** for VEV_5400:

- `inventory_skew_target(t_global)` decays from `+100` at `t=3.0` to `0` at
  `t=5.0`.
- The effective fair used by take/clear/make is
  `BS_theo + skew_strength × (target − current_position)`.
  With `skew_strength = 0.01`, a 100-lot gap shifts effective fair by 1.0
  price unit — enough to cross the typical 1-tick option spread.
- Fair value itself comes from BS at the smile IV curve (not market IV), so
  we already mark 5400 as underpriced even before the skew kicks in.

As the skew target decays, the trader naturally unwinds the long position.

## Parameters to tune

Everything is in `PARAMS` at the top of `trader.py`. Main knobs:

- `fair_ema_alpha` for VELVETFRUIT / HYDROGEL (derived from observed AR(1)
  half-life — changing reshuffles how fast the fair moves with the book).
- `take_width`, `make_width`, `clear_width` per product — larger widths =
  safer but fewer fills.
- `soft_position_limit` per product — triggers the clear layer (cross-book
  reduce) when the inventory gets too large.
- `max_quote_size` caps each side of a make quote so a fat 5400 edge doesn't
  produce 62-lot quotes on a 1-tick spread.
- `max_take_per_tick` caps the take layer, especially for 5400 (20/tick) so
  we don't spend all the edge in one tick.
- `options_common.smile_{a,b,c}` — the fitted smile. If the live-round smile
  shifts, re-fit these.
- `VEV_5400_edge.skew_decay_end_t` — push later if the edge persists longer
  than the linear projection suggests; push earlier if it closes faster.
- `delta_hedge.enabled` — off by default. Turn on and tune `band` if you want
  residual-delta hedging back into VELVETFRUIT.

## Guardrails in place

- Position-limit clipping at every layer (take / clear / make).
- Never quotes zero-sized orders.
- Never crosses own (post-take residual) book.
- Handles missing order depth gracefully — a product with no book just gets
  no orders that tick.
- TTE floored at `1e-6` years; implied-vol solver returns `NaN` (and falls
  back to intrinsic) when the price is outside no-arbitrage bounds.
- `traderData` round-trips EMA state and day counter across ticks; day
  rollover is detected from a timestamp reset.
