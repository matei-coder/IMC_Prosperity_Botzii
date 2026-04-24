"""IMC Prosperity Round 3 — single-file Trader.

Products: VELVETFRUIT_EXTRACT (underlying), HYDROGEL_PACK (independent),
and VEV call options at strikes 4000..6500.

Edge: VEV_5400 prints below the smile fit; hold a long bias that decays over
the 7-day contract life. Everything else is market-made around a fair value
(EMA mid for the two cash assets, BS+smile theoretical for the options).

Submit this file as-is to Prosperity. Locally, `python trader.py` runs the
backtest on CSVs under ./data/.
"""

import json
import math
import os
import csv
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from datamodel import Order, OrderDepth, TradingState, Trade, Listing, Observation


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

PARAMS: dict = {
    "VELVETFRUIT_EXTRACT": {
        "fair_ema_alpha": 0.00347,   # AR(1) half-life ~200 ticks
        "take_width": 1,
        "make_width": 2,
        "clear_width": 0,            # 0 = cross at the opposite best
        "position_limit": 250,
        "soft_position_limit": 200,
        "max_quote_size": 30,
        "max_take_per_tick": 40,
    },
    "HYDROGEL_PACK": {
        "fair_ema_alpha": 0.002,     # AR(1) half-life ~350 ticks
        "take_width": 2,
        "make_width": 4,
        "clear_width": 0,
        "position_limit": 60,
        "soft_position_limit": 40,
        "max_quote_size": 15,
        "max_take_per_tick": 40,
    },
    "options_common": {
        "tte_start_days": 7.0,
        "smile_a": 0.2608,
        "smile_b": 0.0021,
        "smile_c": 0.0284,
        "mispricing_threshold": 0.5,   # price units; doubles as take_width
        "make_width": 1,
        "clear_width": 0,
        "option_position_limit": 200,
        "soft_position_limit": 150,
        "max_quote_size": 25,
        "max_take_per_tick": 40,
    },
    "VEV_5400_edge": {
        "inventory_skew_start": 100,
        "inventory_skew_end": 0,
        "skew_decay_start_t": 3.0,    # t_global (day + ts/1e6)
        "skew_decay_end_t": 5.0,
        "inventory_skew_strength": 0.01,  # price units per unit of (target - pos)
        "aggressive_take_size": 20,
        "max_quote_size": 40,
        "max_take_per_tick": 20,
    },
    "delta_hedge": {
        "enabled": False,
        "band": 30,
    },
    "backtest": {
        "data_dir": "./data/",
        "fill_model": "hybrid_scaled",     # "hybrid_scaled" | "hybrid_flat" | "cross_only"
        "passive_fill_ratio": 0.5,
        "trade_window": 100,               # match trades in [ts, ts+trade_window)
    },
    "debug": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, debug: bool):
        self.debug = debug
        self.buf: List[dict] = []

    def log(self, **kwargs):
        if self.debug:
            self.buf.append(kwargs)

    def flush(self, timestamp: int):
        if self.debug and self.buf:
            print(json.dumps({"t": timestamp, "events": self.buf}))
        self.buf.clear()


LOG = Logger(PARAMS["debug"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Status — safe extractors over a TradingState
# ─────────────────────────────────────────────────────────────────────────────

class Status:
    @staticmethod
    def position(state: TradingState, product: str) -> int:
        return state.position.get(product, 0) if state.position else 0

    @staticmethod
    def depth(state: TradingState, product: str) -> Optional[OrderDepth]:
        if not state.order_depths:
            return None
        return state.order_depths.get(product)

    @staticmethod
    def best_bid(depth: Optional[OrderDepth]) -> Optional[Tuple[int, int]]:
        if depth is None or not depth.buy_orders:
            return None
        p = max(depth.buy_orders.keys())
        return p, depth.buy_orders[p]

    @staticmethod
    def best_ask(depth: Optional[OrderDepth]) -> Optional[Tuple[int, int]]:
        if depth is None or not depth.sell_orders:
            return None
        p = min(depth.sell_orders.keys())
        return p, -depth.sell_orders[p]

    @staticmethod
    def mid(state: TradingState, product: str) -> Optional[float]:
        depth = Status.depth(state, product)
        bb = Status.best_bid(depth)
        ba = Status.best_ask(depth)
        if bb is None and ba is None:
            return None
        if bb is None:
            return float(ba[0])
        if ba is None:
            return float(bb[0])
        return (bb[0] + ba[0]) / 2.0

    @staticmethod
    def spread(state: TradingState, product: str) -> Optional[int]:
        depth = Status.depth(state, product)
        bb = Status.best_bid(depth)
        ba = Status.best_ask(depth)
        if bb is None or ba is None:
            return None
        return ba[0] - bb[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Option math
# ─────────────────────────────────────────────────────────────────────────────

class OptionMath:
    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _npdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def bs_call(S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        st = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / st
        d2 = d1 - st
        return S * OptionMath._ncdf(d1) - K * OptionMath._ncdf(d2)

    @staticmethod
    def delta(S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 1.0 if S > K else 0.0
        st = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / st
        return OptionMath._ncdf(d1)

    @staticmethod
    def vega(S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        st = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / st
        return S * OptionMath._npdf(d1) * math.sqrt(T)

    @staticmethod
    def implied_vol(C: float, S: float, K: float, T: float) -> float:
        intrinsic = max(S - K, 0.0)
        if C < intrinsic - 1e-9 or C >= S or T <= 0:
            return float("nan")
        lo, hi = 1e-4, 5.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            price = OptionMath.bs_call(S, K, T, mid)
            if price < C:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        return 0.5 * (lo + hi)

    @staticmethod
    def smile_iv(S: float, K: float, T: float, params: dict) -> float:
        cc = params["options_common"]
        T_safe = max(T, 1e-6)
        m = math.log(K / S) / math.sqrt(T_safe)
        return cc["smile_a"] + cc["smile_b"] * m + cc["smile_c"] * m * m

    @staticmethod
    def compute_tte_years(day: int, timestamp: int, tte_start_days: float = 7.0) -> float:
        t_global = day + timestamp / 1_000_000.0
        tte_days = tte_start_days - t_global
        return max(tte_days, 1e-6) / 365.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Strategies
# ─────────────────────────────────────────────────────────────────────────────

class Strategy(ABC):
    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params

    @abstractmethod
    def run(self, state: TradingState) -> List[Order]: ...

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass


class MarketMakingStrategy(Strategy):
    """Take → Clear → Make, with position-limit clipping at every step.

    Subclasses provide fair value; optionally an inventory skew target /
    strength that biases the effective fair used by all three layers.
    """

    def __init__(self, symbol: str, params: dict):
        super().__init__(symbol, params)
        self.cfg = self._build_cfg()

    def _build_cfg(self) -> dict:
        return dict(self.params[self.symbol])

    @abstractmethod
    def get_fair_value(self, state: TradingState) -> Optional[float]: ...

    def get_skew_target(self, state: TradingState) -> int:
        return 0

    def get_skew_strength(self) -> float:
        return 0.0

    def get_take_width(self) -> float:
        return self.cfg["take_width"]

    def run(self, state: TradingState) -> List[Order]:
        depth = Status.depth(state, self.symbol)
        if depth is None or (not depth.buy_orders and not depth.sell_orders):
            return []

        fair = self.get_fair_value(state)
        if fair is None or not math.isfinite(fair):
            return []

        position = Status.position(state, self.symbol)
        skew_target = self.get_skew_target(state)
        skew_strength = self.get_skew_strength()
        eff_fair = fair + skew_strength * (skew_target - position)

        limit = self.cfg["position_limit"]
        buy_cap = limit - position
        sell_cap = limit + position

        # Mutable copies of book liquidity — consumed sequentially by take/clear
        asks = {p: -q for p, q in depth.sell_orders.items()}   # price -> positive qty
        bids = {p: q for p, q in depth.buy_orders.items()}     # price -> positive qty

        orders: List[Order] = []

        # ── TAKE ──
        t_orders, buy_used, sell_used = self._take(eff_fair, asks, bids, buy_cap, sell_cap)
        orders.extend(t_orders)
        buy_cap -= buy_used
        sell_cap -= sell_used
        position += buy_used - sell_used

        # ── CLEAR ──
        c_orders, buy_used, sell_used = self._clear(position, asks, bids, buy_cap, sell_cap)
        orders.extend(c_orders)
        buy_cap -= buy_used
        sell_cap -= sell_used
        position += buy_used - sell_used

        # ── MAKE ──
        m_orders = self._make(eff_fair, asks, bids, position, buy_cap, sell_cap)
        orders.extend(m_orders)

        LOG.log(sym=self.symbol, fair=round(fair, 3), eff=round(eff_fair, 3),
                pos=position, n=len(orders))
        return orders

    # --- take: walk all levels inside the threshold, up to max_take_per_tick ---
    def _take(self, eff_fair: float, asks: dict, bids: dict,
              buy_cap: int, sell_cap: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        tw = self.get_take_width()
        cap_take = self.cfg["max_take_per_tick"]
        buy_used = 0
        sell_used = 0

        # Lift asks with price <= eff_fair - tw
        for p in sorted(asks.keys()):
            if p > eff_fair - tw:
                break
            free = min(buy_cap - buy_used, cap_take - buy_used)
            if free <= 0:
                break
            take = min(asks[p], free)
            if take <= 0:
                continue
            orders.append(Order(self.symbol, int(p), int(take)))
            asks[p] -= take
            buy_used += take

        # Hit bids with price >= eff_fair + tw
        for p in sorted(bids.keys(), reverse=True):
            if p < eff_fair + tw:
                break
            free = min(sell_cap - sell_used, cap_take - sell_used)
            if free <= 0:
                break
            take = min(bids[p], free)
            if take <= 0:
                continue
            orders.append(Order(self.symbol, int(p), -int(take)))
            bids[p] -= take
            sell_used += take

        return orders, buy_used, sell_used

    # --- clear: cross the book when |position| > soft_position_limit ---
    def _clear(self, position: int, asks: dict, bids: dict,
               buy_cap: int, sell_cap: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        soft = self.cfg["soft_position_limit"]
        cw = self.cfg["clear_width"]
        buy_used = 0
        sell_used = 0

        if position > soft:
            excess = position - soft
            for p in sorted(bids.keys(), reverse=True):
                if excess <= 0:
                    break
                free = sell_cap - sell_used
                if free <= 0:
                    break
                take = min(bids[p], excess, free)
                if take <= 0:
                    continue
                price = int(p - cw)  # cw=0 → exactly at the bid
                orders.append(Order(self.symbol, price, -int(take)))
                bids[p] -= take
                sell_used += take
                excess -= take
        elif position < -soft:
            excess = -position - soft
            for p in sorted(asks.keys()):
                if excess <= 0:
                    break
                free = buy_cap - buy_used
                if free <= 0:
                    break
                take = min(asks[p], excess, free)
                if take <= 0:
                    continue
                price = int(p + cw)
                orders.append(Order(self.symbol, price, int(take)))
                asks[p] -= take
                buy_used += take
                excess -= take

        return orders, buy_used, sell_used

    # --- make: passive quotes both sides, sized from free capacity ---
    def _make(self, eff_fair: float, asks: dict, bids: dict, position: int,
              buy_cap: int, sell_cap: int) -> List[Order]:
        """`asks`/`bids` are the post-take book (prices with remaining volume > 0)."""
        orders: List[Order] = []
        mw = self.cfg["make_width"]
        max_qs = self.cfg["max_quote_size"]
        limit = self.cfg["position_limit"]

        base_size = max(1, (limit - abs(position)) // 4)
        size = min(max_qs, base_size)

        bid_price = int(math.floor(eff_fair - mw))
        ask_price = int(math.ceil(eff_fair + mw))

        # Post-take residual top of book
        rem_asks = [p for p, v in asks.items() if v > 0]
        rem_bids = [p for p, v in bids.items() if v > 0]
        post_best_ask = min(rem_asks) if rem_asks else None
        post_best_bid = max(rem_bids) if rem_bids else None

        # Never cross what's left of the book.
        if post_best_ask is not None and bid_price >= post_best_ask:
            bid_price = post_best_ask - 1
        if post_best_bid is not None and ask_price <= post_best_bid:
            ask_price = post_best_bid + 1

        if ask_price <= bid_price:
            return orders

        bid_qty = min(size, max(0, buy_cap))
        ask_qty = min(size, max(0, sell_cap))

        if bid_qty > 0 and bid_price > 0:
            orders.append(Order(self.symbol, bid_price, int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(self.symbol, ask_price, -int(ask_qty)))

        return orders


class MeanReversionMM(MarketMakingStrategy):
    """Fair = EMA of mid prices, persisted in traderData."""

    def __init__(self, symbol: str, params: dict):
        super().__init__(symbol, params)
        self.ema: Optional[float] = None

    def save(self) -> dict:
        return {"ema": self.ema}

    def load(self, data: dict) -> None:
        v = data.get("ema")
        self.ema = float(v) if v is not None else None

    def get_fair_value(self, state: TradingState) -> Optional[float]:
        mid = Status.mid(state, self.symbol)
        if mid is None:
            return self.ema
        alpha = self.cfg["fair_ema_alpha"]
        if self.ema is None:
            self.ema = mid
        else:
            self.ema = alpha * mid + (1 - alpha) * self.ema
        return self.ema


class OptionMM(MarketMakingStrategy):
    """Fair = BS theoretical at smile IV. Optional dynamic inventory skew."""

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    def __init__(self, symbol: str, strike: int, params: dict, skew: Union[str, int] = 0):
        self.strike = strike
        self.skew_mode = skew                                # 0 or "dynamic"
        super().__init__(symbol, params)

    def _build_cfg(self) -> dict:
        base = self.params["options_common"]
        cfg = {
            "take_width": base["mispricing_threshold"],
            "make_width": base["make_width"],
            "clear_width": base["clear_width"],
            "position_limit": base["option_position_limit"],
            "soft_position_limit": base["soft_position_limit"],
            "max_quote_size": base["max_quote_size"],
            "max_take_per_tick": base["max_take_per_tick"],
        }
        if self.symbol == "VEV_5400":
            edge = self.params["VEV_5400_edge"]
            cfg["max_quote_size"] = edge["max_quote_size"]
            cfg["max_take_per_tick"] = edge["max_take_per_tick"]
        return cfg

    def get_fair_value(self, state: TradingState) -> Optional[float]:
        S = Status.mid(state, self.UNDERLYING)
        if S is None:
            return None
        day = getattr(state, "day", 0)
        T = OptionMath.compute_tte_years(
            day, state.timestamp,
            self.params["options_common"]["tte_start_days"],
        )
        iv = OptionMath.smile_iv(S, self.strike, T, self.params)
        return OptionMath.bs_call(S, self.strike, T, iv)

    def get_skew_target(self, state: TradingState) -> int:
        if self.skew_mode != "dynamic":
            return 0
        day = getattr(state, "day", 0)
        t_global = day + state.timestamp / 1_000_000.0
        cfg = self.params["VEV_5400_edge"]
        ts, te = cfg["skew_decay_start_t"], cfg["skew_decay_end_t"]
        ss, se = cfg["inventory_skew_start"], cfg["inventory_skew_end"]
        if t_global <= ts:
            return ss
        if t_global >= te:
            return se
        frac = (t_global - ts) / (te - ts)
        return int(round(ss + frac * (se - ss)))

    def get_skew_strength(self) -> float:
        if self.skew_mode != "dynamic":
            return 0.0
        return self.params["VEV_5400_edge"]["inventory_skew_strength"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Delta hedger (off by default)
# ─────────────────────────────────────────────────────────────────────────────

class DeltaHedger:
    OPTION_STRIKES = {
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    }
    UNDERLYING = "VELVETFRUIT_EXTRACT"

    def __init__(self, params: dict):
        self.params = params

    def hedge(self, state: TradingState, pending_velvet: List[Order]) -> List[Order]:
        band = self.params["delta_hedge"]["band"]
        S = Status.mid(state, self.UNDERLYING)
        if S is None:
            return []
        day = getattr(state, "day", 0)
        T = OptionMath.compute_tte_years(
            day, state.timestamp,
            self.params["options_common"]["tte_start_days"],
        )

        net = 0.0
        for sym, K in self.OPTION_STRIKES.items():
            pos = Status.position(state, sym)
            if pos == 0:
                continue
            iv = OptionMath.smile_iv(S, K, T, self.params)
            net += pos * OptionMath.delta(S, K, T, iv)

        velvet_pos = Status.position(state, self.UNDERLYING) + sum(o.quantity for o in pending_velvet)
        total = net + velvet_pos

        if abs(total) <= band:
            return []

        qty = -int(round(total))
        limit = self.params[self.UNDERLYING]["position_limit"]
        depth = Status.depth(state, self.UNDERLYING)
        if depth is None:
            return []

        if qty > 0:
            qty = min(qty, limit - velvet_pos)
            if qty <= 0:
                return []
            ba = Status.best_ask(depth)
            if ba is None:
                return []
            return [Order(self.UNDERLYING, ba[0], qty)]
        else:
            qty = max(qty, -(limit + velvet_pos))
            if qty >= 0:
                return []
            bb = Status.best_bid(depth)
            if bb is None:
                return []
            return [Order(self.UNDERLYING, bb[0], qty)]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    def __init__(self):
        self.params = PARAMS
        self.strategies: Dict[str, Strategy] = {
            "VELVETFRUIT_EXTRACT": MeanReversionMM("VELVETFRUIT_EXTRACT", PARAMS),
            "HYDROGEL_PACK":       MeanReversionMM("HYDROGEL_PACK", PARAMS),
            "VEV_5000": OptionMM("VEV_5000", 5000, PARAMS, skew=0),
            "VEV_5100": OptionMM("VEV_5100", 5100, PARAMS, skew=0),
            "VEV_5200": OptionMM("VEV_5200", 5200, PARAMS, skew=0),
            "VEV_5300": OptionMM("VEV_5300", 5300, PARAMS, skew=0),
            "VEV_5400": OptionMM("VEV_5400", 5400, PARAMS, skew="dynamic"),
            "VEV_5500": OptionMM("VEV_5500", 5500, PARAMS, skew=0),
        }
        self.delta_hedger = DeltaHedger(PARAMS) if PARAMS["delta_hedge"]["enabled"] else None
        self.day = 0
        self.last_ts = -1

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # ── load state ──
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}
            self.day = int(td.get("_day", 0))
            self.last_ts = int(td.get("_last_ts", -1))
            if self.last_ts >= 0 and state.timestamp < self.last_ts:
                self.day += 1
            for sym, strat in self.strategies.items():
                if sym in td and isinstance(td[sym], dict):
                    strat.load(td[sym])

        # Attach day so option strategies can compute TTE.
        state.day = self.day  # type: ignore[attr-defined]

        # ── run strategies ──
        all_orders: Dict[str, List[Order]] = {}
        for sym, strat in self.strategies.items():
            out = strat.run(state)
            if out:
                all_orders.setdefault(sym, []).extend(out)

        # ── delta hedge (optional) ──
        if self.delta_hedger is not None:
            hedge = self.delta_hedger.hedge(state, all_orders.get("VELVETFRUIT_EXTRACT", []))
            if hedge:
                all_orders.setdefault("VELVETFRUIT_EXTRACT", []).extend(hedge)

        # ── save state ──
        td_out = {"_day": self.day, "_last_ts": state.timestamp}
        for sym, strat in self.strategies.items():
            td_out[sym] = strat.save()
        trader_data = json.dumps(td_out)

        LOG.flush(state.timestamp)
        return all_orders, 0, trader_data


# ─────────────────────────────────────────────────────────────────────────────
# 8. Backtester
# ─────────────────────────────────────────────────────────────────────────────

def _parse_prices_csv(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            rows.append(r)
    return rows


def _parse_trades_csv(path: str, day: int) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
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


def _simulate_fills(orders: List[Order], depth: OrderDepth,
                    trades_in_window: List[dict], params: dict) -> List[Tuple[str, int, float, int]]:
    """Return list of (symbol, signed_qty, price, _pad) representing fills.

    Signed qty: positive for buys, negative for sells.
    """
    fills: List[Tuple[str, int, float, int]] = []
    fm = params["backtest"]["fill_model"]
    ratio = params["backtest"]["passive_fill_ratio"]

    for order in orders:
        sym = order.symbol
        q = order.quantity
        p = order.price
        if q == 0:
            continue

        if q > 0:  # buy
            remaining = q
            # Cross-book: asks at price <= p
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p > p or remaining <= 0:
                    break
                vol = -depth.sell_orders[ask_p]
                take = min(vol, remaining)
                if take > 0:
                    fills.append((sym, take, float(ask_p), 0))
                    depth.sell_orders[ask_p] += take  # less negative
                    if depth.sell_orders[ask_p] == 0:
                        del depth.sell_orders[ask_p]
                    remaining -= take
            # Passive (hybrid)
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
                        fills.append((sym, fill_qty, float(p), 0))
                        remaining -= fill_qty
        else:  # sell
            remaining = -q
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p < p or remaining <= 0:
                    break
                vol = depth.buy_orders[bid_p]
                take = min(vol, remaining)
                if take > 0:
                    fills.append((sym, -take, float(bid_p), 0))
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
                        fills.append((sym, -fill_qty, float(p), 0))
                        remaining -= fill_qty

    return fills


def run_backtest(data_dir: str) -> None:
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        print(f"[backtest] data dir not found: {data_dir}")
        print(f"[backtest] put prices_round_3_day_{{0,1,2}}.csv + trades_*.csv under {data_dir}/ to run")
        return

    days = [0, 1, 2]
    prices_by_day: Dict[int, List[dict]] = {}
    trades_by_day: Dict[int, List[dict]] = {}
    for d in days:
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
    per_product_pnl_series: Dict[str, List[float]] = defaultdict(list)
    per_product_cash: Dict[str, float] = defaultdict(float)

    window = PARAMS["backtest"]["trade_window"]

    for d in sorted(prices_by_day.keys()):
        rows = prices_by_day[d]
        trades = sorted(trades_by_day.get(d, []), key=lambda x: x["timestamp"])

        # Group prices by timestamp
        rows_by_ts: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            rows_by_ts[int(r["timestamp"])].append(r)
        timestamps = sorted(rows_by_ts.keys())

        # Index trades by timestamp for window lookup
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
                od = _build_order_depth(r)
                order_depths[prod] = od
                try:
                    mids_at_ts[prod] = float(r.get("mid_price") or 0.0)
                except Exception:
                    pass

            # Gather historical trades in [ts, ts+window)
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

            # Flatten orders, simulate fills
            tick_fills: List[Tuple[str, int, float, int]] = []
            for sym, olist in orders_by_sym.items():
                if sym not in order_depths:
                    continue
                tick_fills.extend(_simulate_fills(olist, order_depths[sym], window_trades, PARAMS))

            # Apply fills
            new_own_trades: Dict[str, List[Trade]] = defaultdict(list)
            for sym, qty, price, _ in tick_fills:
                positions[sym] += qty
                cash -= qty * price                # buy → cash out; sell → cash in
                per_product_cash[sym] -= qty * price
                trade_counts[sym] += 1
                new_own_trades[sym].append(
                    Trade(sym, int(round(price)), qty, "SELF", "MARKET", ts)
                )

                # Enforce limits (sanity check)
                if sym in PARAMS:
                    lim = PARAMS[sym]["position_limit"]
                elif sym.startswith("VEV_"):
                    lim = PARAMS["options_common"]["option_position_limit"]
                else:
                    lim = 10**9
                if abs(positions[sym]) > lim:
                    print(f"[WARN] limit breach {sym} pos={positions[sym]} lim={lim} ts={ts} day={d}")

            prev_own_trades = new_own_trades

            # MtM snapshot per product (for sharpe-like)
            for sym in list(per_product_cash.keys()) + list(positions.keys()):
                mid = mids_at_ts.get(sym)
                if mid is None:
                    continue
                pnl = per_product_cash[sym] + positions[sym] * mid
                per_product_pnl_series[sym].append(pnl)

        per_day_cash[d] = cash - day_start_cash

    # ── summary ──
    print("\n──────── Backtest summary ────────")
    print(f"Days: {sorted(prices_by_day.keys())}   fill_model: {PARAMS['backtest']['fill_model']}")

    # Final MtM using last observed mids
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
    print(f"\n{'product':<22} {'trades':>7} {'finalpos':>9} {'cash':>14} {'MtM':>14} {'sharpe':>8}")
    all_syms = sorted(set(list(positions.keys()) + list(per_product_cash.keys())))
    for sym in all_syms:
        pos = positions[sym]
        pc = per_product_cash[sym]
        mid = last_mids.get(sym, 0.0)
        mtm = pc + pos * mid
        total_mtm += mtm
        series = per_product_pnl_series.get(sym, [])
        sharpe = _series_sharpe(series)
        print(f"{sym:<22} {trade_counts[sym]:>7} {pos:>9} {pc:>14.2f} {mtm:>14.2f} {sharpe:>8.3f}")
    print(f"\n{'TOTAL':<22} {'':>7} {'':>9} {cash:>14.2f} {total_mtm:>14.2f}")

    print("\nPer-day cash delta:")
    for d, v in per_day_cash.items():
        print(f"  day {d}: {v:+.2f}")


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
# 9. Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_backtest(PARAMS["backtest"]["data_dir"])
