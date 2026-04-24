"""
trader_ipr.py — Directional pe INTARIAN_PEPPER_ROOT
Acumuleaza max long pana la ts=90000, apoi vinde tot.
"""

import json
from typing import Any, Dict, List
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)

# ── logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json([
                self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                self.compress_orders(orders),
                conversions,
                self.truncate(trader_data, max_item_length),
                self.truncate(self.logs, max_item_length),
            ])
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol, trade.price, trade.quantity,
                    trade.buyer, trade.seller, trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice,
                observation.transportFees, observation.exportTariff,
                observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()

# ── constante ─────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
SELL_TS_START  = 90_000   # switch la sell phase

# ── trader ────────────────────────────────────────────────────────────────────

class Trader:

    def __init__(self):
        self.current_day: int = 0
        self.prev_ts: int     = 0

    # ── persistenta pipe-separated ───────────────────────────────────────────
    # format: "current_day|prev_timestamp"

    def _load(self, raw: str, timestamp: int):
        if not raw:
            self.current_day = 0
            self.prev_ts     = timestamp
            return
        parts            = raw.split("|")
        self.current_day = int(parts[0])
        self.prev_ts     = int(parts[1])
        if timestamp < self.prev_ts:
            self.current_day += 1
        self.prev_ts = timestamp

    def _save(self) -> str:
        return f"{self.current_day}|{self.prev_ts}"

    def run(self, state: TradingState):
        self._load(state.traderData, state.timestamp)
        orders: Dict[str, List[Order]] = {}

        pos_ipr = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        od = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if od is not None:
            orders["INTARIAN_PEPPER_ROOT"] = self._ipr(od, pos_ipr, state.timestamp)

        trader_data = self._save()
        logger.flush(state, orders, 0, trader_data)
        return orders, 0, trader_data

    # ── INTARIAN_PEPPER_ROOT — directional ───────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        result: List[Order] = []

        bids = sorted(od.buy_orders.items(), reverse=True)
        asks = sorted(od.sell_orders.items())

        if timestamp >= SELL_TS_START:
            # sell phase: vinde pana la 0 (nu merge short), cu price floor dinamic
            fv = 12_000 + 1_000 * self.current_day + 0.001 * timestamp
            for bid, bid_vol in bids:
                sell_cap = pos
                if sell_cap <= 0:
                    break
                vol = int(min(bid_vol, sell_cap))
                if vol > 0:
                    limit_sell = max(bid, round(fv) - 2)
                    result.append(Order("INTARIAN_PEPPER_ROOT", limit_sell, -vol))
                    pos -= vol
        else:
            # accumulation phase: lift every ask until max long
            for ask, ask_vol in asks:
                buy_cap = POSITION_LIMIT - pos
                if buy_cap <= 0:
                    break
                vol = int(min(abs(ask_vol), buy_cap))
                if vol > 0:
                    result.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                    pos += vol
            # passive bid pentru capacitatea ramasa
            if pos < POSITION_LIMIT and asks:
                buy_cap = POSITION_LIMIT - pos
                result.append(Order("INTARIAN_PEPPER_ROOT", asks[0][0] - 1, buy_cap))

        return result
