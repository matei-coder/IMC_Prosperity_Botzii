"""Local backtest harness for trader.py. NEVER submit this file.

Loads Round 3 price + trade CSVs from ./data/, reconstructs TradingState tick
by tick, runs Trader, and simulates fills under a selectable model.

Run:
    python3 backtest.py                     # uses ./data/
    python3 backtest.py /path/to/data       # custom dir
"""

import os
import sys
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState, Trade, Listing, Observation
from trader import Trader, PARAMS


# ── backtest config (kept out of trader.py's PARAMS since it's local-only) ──
BACKTEST_CFG = {
    "data_dir": "./data/",
    "fill_model": "hybrid_scaled",   # "hybrid_scaled" | "hybrid_flat" | "cross_only"
    "passive_fill_ratio": 0.5,
    "trade_window": 100,
    "days": [0, 1, 2],
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_prices_csv(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(r)
    return rows


def _parse_trades_csv(path: str, day: int) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                rows.append({
                    "day": day,
                    "timestamp": int(r["timestamp"]),
                    "symbol": r["symbol"],
                    "price": float(r["price"]),
                    "quantity": int(r["quantity"]),
                })
            except Exception:
                continue
    return rows


def _build_order_depth(row: dict) -> OrderDepth:
    d = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if bp and bv and bp != "" and bv != "":
            try:
                d.buy_orders[int(float(bp))] = int(float(bv))
            except Exception:
                pass
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if ap and av and ap != "" and av != "":
            try:
                d.sell_orders[int(float(ap))] = -int(float(av))
            except Exception:
                pass
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Fill simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_fills(orders: List[Order], depth: OrderDepth,
                    trades_in_window: List[dict],
                    cfg: dict) -> List[Tuple[str, int, float, str]]:
    """Return [(symbol, signed_qty, price, kind), ...] where kind ∈ {"agg","pas"}."""
    fills: List[Tuple[str, int, float, str]] = []
    fm = cfg["fill_model"]
    ratio = cfg["passive_fill_ratio"]

    for order in orders:
        sym = order.symbol
        q = order.quantity
        p = order.price
        if q == 0:
            continue

        if q > 0:  # buy
            remaining = q
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p > p or remaining <= 0:
                    break
                vol = -depth.sell_orders[ask_p]
                take = min(vol, remaining)
                if take > 0:
                    fills.append((sym, take, float(ask_p), "agg"))
                    depth.sell_orders[ask_p] += take
                    if depth.sell_orders[ask_p] == 0:
                        del depth.sell_orders[ask_p]
                    remaining -= take
            if remaining > 0 and fm.startswith("hybrid"):
                for tr in trades_in_window:
                    if tr["symbol"] != sym or remaining <= 0:
                        continue
                    if p < tr["price"]:
                        continue
                    if fm == "hybrid_scaled":
                        r = 1.0 if p > tr["price"] else ratio
                    else:
                        r = ratio
                    fill_qty = min(remaining, max(1, int(tr["quantity"] * r)))
                    if fill_qty > 0:
                        fills.append((sym, fill_qty, float(p), "pas"))
                        remaining -= fill_qty
        else:  # sell
            remaining = -q
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p < p or remaining <= 0:
                    break
                vol = depth.buy_orders[bid_p]
                take = min(vol, remaining)
                if take > 0:
                    fills.append((sym, -take, float(bid_p), "agg"))
                    depth.buy_orders[bid_p] -= take
                    if depth.buy_orders[bid_p] == 0:
                        del depth.buy_orders[bid_p]
                    remaining -= take
            if remaining > 0 and fm.startswith("hybrid"):
                for tr in trades_in_window:
                    if tr["symbol"] != sym or remaining <= 0:
                        continue
                    if p > tr["price"]:
                        continue
                    if fm == "hybrid_scaled":
                        r = 1.0 if p < tr["price"] else ratio
                    else:
                        r = ratio
                    fill_qty = min(remaining, max(1, int(tr["quantity"] * r)))
                    if fill_qty > 0:
                        fills.append((sym, -fill_qty, float(p), "pas"))
                        remaining -= fill_qty
    return fills


def _series_sharpe(series: List[float]) -> float:
    if len(series) < 2:
        return 0.0
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(n - 1, 1)
    std = math.sqrt(var)
    if std <= 1e-9:
        return 0.0
    return (mean / std) * math.sqrt(n)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(data_dir: str, cfg: dict = BACKTEST_CFG) -> None:
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        print(f"[backtest] data dir not found: {data_dir}")
        print(f"[backtest] put prices_round_3_day_{{0,1,2}}.csv + trades_*.csv there")
        return

    prices_by_day: Dict[int, List[dict]] = {}
    trades_by_day: Dict[int, List[dict]] = {}
    for d in cfg["days"]:
        p_path = os.path.join(data_dir, f"prices_round_3_day_{d}.csv")
        t_path = os.path.join(data_dir, f"trades_round_3_day_{d}.csv")
        if not os.path.exists(p_path):
            print(f"[backtest] missing {p_path}; skipping day {d}")
            continue
        prices_by_day[d] = _parse_prices_csv(p_path)
        trades_by_day[d] = _parse_trades_csv(t_path, d)

    if not prices_by_day:
        print("[backtest] no data to run")
        return

    trader = Trader()
    trader_data = ""

    cash = 0.0
    positions: Dict[str, int] = defaultdict(int)
    trade_counts: Dict[str, int] = defaultdict(int)
    per_day_cash = {d: 0.0 for d in prices_by_day}
    per_day_product_cash: Dict[int, Dict[str, float]] = {d: defaultdict(float) for d in prices_by_day}
    per_day_product_pos_start: Dict[int, Dict[str, int]] = {d: {} for d in prices_by_day}
    per_product_pnl_series: Dict[str, List[float]] = defaultdict(list)
    per_product_cash: Dict[str, float] = defaultdict(float)
    agg_fills: Dict[str, int] = defaultdict(int)
    pas_fills: Dict[str, int] = defaultdict(int)
    both_side_ticks: Dict[str, int] = defaultdict(int)
    buy_count: Dict[str, int] = defaultdict(int)
    sell_count: Dict[str, int] = defaultdict(int)
    buy_value: Dict[str, float] = defaultdict(float)   # sum of (price * qty) for buys
    sell_value: Dict[str, float] = defaultdict(float)  # sum of (price * qty) for sells
    buy_qty_sum: Dict[str, int] = defaultdict(int)
    sell_qty_sum: Dict[str, int] = defaultdict(int)
    # Day-scoped copies for the day-2 report
    day_buy_count: Dict[int, Dict[str, int]] = {d: defaultdict(int) for d in prices_by_day}
    day_sell_count: Dict[int, Dict[str, int]] = {d: defaultdict(int) for d in prices_by_day}
    day_agg_fills: Dict[int, Dict[str, int]] = {d: defaultdict(int) for d in prices_by_day}
    day_pas_fills: Dict[int, Dict[str, int]] = {d: defaultdict(int) for d in prices_by_day}
    day_buy_value: Dict[int, Dict[str, float]] = {d: defaultdict(float) for d in prices_by_day}
    day_sell_value: Dict[int, Dict[str, float]] = {d: defaultdict(float) for d in prices_by_day}
    day_buy_qty_sum: Dict[int, Dict[str, int]] = {d: defaultdict(int) for d in prices_by_day}
    day_sell_qty_sum: Dict[int, Dict[str, int]] = {d: defaultdict(int) for d in prices_by_day}

    window = cfg["trade_window"]

    for d in sorted(prices_by_day.keys()):
        rows = prices_by_day[d]
        trades = sorted(trades_by_day.get(d, []), key=lambda x: x["timestamp"])

        rows_by_ts: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            rows_by_ts[int(r["timestamp"])].append(r)
        timestamps = sorted(rows_by_ts.keys())

        trades_by_ts: Dict[int, List[dict]] = defaultdict(list)
        for tr in trades:
            trades_by_ts[tr["timestamp"]].append(tr)

        day_start_cash = cash
        listings = {r["product"]: Listing(r["product"], r["product"], "SEASHELLS") for r in rows}
        prev_own_trades: Dict[str, List[Trade]] = defaultdict(list)

        for ts in timestamps:
            order_depths: Dict[str, OrderDepth] = {}
            mids_at_ts: Dict[str, float] = {}
            for r in rows_by_ts[ts]:
                prod = r["product"]
                order_depths[prod] = _build_order_depth(r)
                try:
                    mids_at_ts[prod] = float(r.get("mid_price") or 0.0)
                except Exception:
                    pass

            window_trades: List[dict] = []
            for t_look in range(ts, ts + window, 100):
                if t_look in trades_by_ts:
                    window_trades.extend(trades_by_ts[t_look])

            state = TradingState(
                traderData=trader_data,
                timestamp=ts,
                listings=listings,
                order_depths=order_depths,
                own_trades=prev_own_trades,
                market_trades={},
                position=dict(positions),
                observations=Observation({}, {}),
            )

            orders_by_sym, _conv, trader_data = trader.run(state)

            # Track same-tick both-sides quoting (before any clipping/fill logic).
            for sym, olist in orders_by_sym.items():
                has_buy = any(o.quantity > 0 for o in olist)
                has_sell = any(o.quantity < 0 for o in olist)
                if has_buy and has_sell:
                    both_side_ticks[sym] += 1

            tick_fills: List[Tuple[str, int, float, str]] = []
            for sym, olist in orders_by_sym.items():
                if sym not in order_depths:
                    continue
                tick_fills.extend(_simulate_fills(olist, order_depths[sym], window_trades, cfg))

            new_own_trades: Dict[str, List[Trade]] = defaultdict(list)
            for sym, qty, price, kind in tick_fills:
                positions[sym] += qty
                cash -= qty * price
                per_product_cash[sym] -= qty * price
                per_day_product_cash[d][sym] -= qty * price
                trade_counts[sym] += 1
                if kind == "agg":
                    agg_fills[sym] += 1
                    day_agg_fills[d][sym] += 1
                else:
                    pas_fills[sym] += 1
                    day_pas_fills[d][sym] += 1
                if qty > 0:
                    buy_count[sym] += 1
                    buy_value[sym] += price * qty
                    buy_qty_sum[sym] += qty
                    day_buy_count[d][sym] += 1
                    day_buy_value[d][sym] += price * qty
                    day_buy_qty_sum[d][sym] += qty
                else:
                    sell_count[sym] += 1
                    sell_value[sym] += price * (-qty)
                    sell_qty_sum[sym] += (-qty)
                    day_sell_count[d][sym] += 1
                    day_sell_value[d][sym] += price * (-qty)
                    day_sell_qty_sum[d][sym] += (-qty)
                new_own_trades[sym].append(
                    Trade(sym, int(round(price)), qty, "SELF", "MARKET", ts)
                )
                lim = PARAMS[sym]["position_limit"] if sym in PARAMS \
                    else PARAMS["options_common"]["option_position_limit"]
                if abs(positions[sym]) > lim:
                    print(f"[WARN] limit breach {sym} pos={positions[sym]} lim={lim} ts={ts} day={d}")

            prev_own_trades = new_own_trades

            for sym in set(list(per_product_cash.keys()) + list(positions.keys())):
                mid = mids_at_ts.get(sym)
                if mid is None:
                    continue
                per_product_pnl_series[sym].append(per_product_cash[sym] + positions[sym] * mid)

        per_day_cash[d] = cash - day_start_cash

    # ── summary ──
    print("\n──────── Backtest summary ────────")
    print(f"Days: {sorted(prices_by_day.keys())}   fill_model: {cfg['fill_model']}")

    last_mids: Dict[str, float] = {}
    for d in sorted(prices_by_day.keys(), reverse=True):
        for r in reversed(prices_by_day[d]):
            prod = r["product"]
            if prod not in last_mids:
                try:
                    last_mids[prod] = float(r.get("mid_price") or 0.0)
                except Exception:
                    pass

    total_mtm = 0.0
    print(f"\n{'product':<22} {'trades':>7} {'buys':>5} {'sells':>5} {'finalpos':>9} "
          f"{'cash':>14} {'MtM':>14} {'agg%':>6} {'pas%':>6} {'avg_edge':>9}")
    all_syms = sorted(set(list(positions.keys()) + list(per_product_cash.keys())))
    for sym in all_syms:
        pos = positions[sym]
        pc = per_product_cash[sym]
        mid = last_mids.get(sym, 0.0)
        mtm = pc + pos * mid
        total_mtm += mtm
        a = agg_fills[sym]
        p_ = pas_fills[sym]
        total_f = a + p_
        agg_pct = (100.0 * a / total_f) if total_f else 0.0
        pas_pct = (100.0 * p_ / total_f) if total_f else 0.0
        avg_buy = (buy_value[sym] / buy_qty_sum[sym]) if buy_qty_sum[sym] else 0.0
        avg_sell = (sell_value[sym] / sell_qty_sum[sym]) if sell_qty_sum[sym] else 0.0
        avg_edge = (avg_sell - avg_buy) if (buy_qty_sum[sym] and sell_qty_sum[sym]) else 0.0
        print(f"{sym:<22} {trade_counts[sym]:>7} {buy_count[sym]:>5} {sell_count[sym]:>5} "
              f"{pos:>9} {pc:>14.2f} {mtm:>14.2f} "
              f"{agg_pct:>5.1f}% {pas_pct:>5.1f}% {avg_edge:>+9.3f}")
    print(f"\n{'TOTAL':<22} {'':>7} {'':>5} {'':>5} {'':>9} {cash:>14.2f} {total_mtm:>14.2f}")

    print("\nPer-day cash delta:")
    for d, v in per_day_cash.items():
        print(f"  day {d}: {v:+.2f}")

    # ── Day-2 focused report ──
    if 2 in prices_by_day:
        print("\n──────── Day 2 report ────────")
        print(f"Total day-2 cash delta: {per_day_cash[2]:+.2f}")
        print(f"\n{'product':<22} {'trades':>7} {'buys':>5} {'sells':>5} "
              f"{'cash_d2':>14} {'agg%':>6} {'pas%':>6} {'avg_edge':>9}")
        day_total_syms = sorted(set(list(day_buy_count[2].keys()) + list(day_sell_count[2].keys())
                                    + list(per_day_product_cash[2].keys())))
        for sym in day_total_syms:
            a = day_agg_fills[2][sym]
            p_ = day_pas_fills[2][sym]
            total_f = a + p_
            agg_pct = (100.0 * a / total_f) if total_f else 0.0
            pas_pct = (100.0 * p_ / total_f) if total_f else 0.0
            bq = day_buy_qty_sum[2][sym]
            sq = day_sell_qty_sum[2][sym]
            avg_buy = (day_buy_value[2][sym] / bq) if bq else 0.0
            avg_sell = (day_sell_value[2][sym] / sq) if sq else 0.0
            avg_edge = (avg_sell - avg_buy) if (bq and sq) else 0.0
            cd = per_day_product_cash[2][sym]
            bc = day_buy_count[2][sym]
            sc = day_sell_count[2][sym]
            print(f"{sym:<22} {bc+sc:>7} {bc:>5} {sc:>5} {cd:>+14.2f} "
                  f"{agg_pct:>5.1f}% {pas_pct:>5.1f}% {avg_edge:>+9.3f}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else BACKTEST_CFG["data_dir"]
    run_backtest(d)
