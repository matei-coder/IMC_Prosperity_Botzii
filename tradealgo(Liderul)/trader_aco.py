"""
trader_aco.py — Market making pe ASH_COATED_OSMIUM
"""

import json
from typing import Any, Dict, List, Optional
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

POSITION_LIMIT       = 80
ACO_FV               = 10_000
ACO_EMA_ALPHA        = 0.11
ACO_SPREAD           = 18
ACO_MAX_VOL          = 20
ACO_TAKE_EDGE        = 4
ACO_SKEW_FACTOR      = 0.10
MOMENTUM_SKEW_FACTOR = 0.3
IMBALANCE_FACTOR     = 1.5

# ── trader ────────────────────────────────────────────────────────────────────

class Trader:

    def __init__(self):
        self.aco_ema: float                = ACO_FV
        self.prev_mid_aco: Optional[float] = None

    # ── persistenta pipe-separated ───────────────────────────────────────────
    # format: "aco_ema|prev_mid_aco"

    def _load(self, raw: str):
        if not raw:
            self.aco_ema      = float(ACO_FV)
            self.prev_mid_aco = None
            return
        parts             = raw.split("|")
        self.aco_ema      = float(parts[0])
        self.prev_mid_aco = float(parts[1]) if len(parts) > 1 and parts[1] != "None" else None

    def _save(self) -> str:
        mid = "None" if self.prev_mid_aco is None else str(self.prev_mid_aco)
        return f"{self.aco_ema}|{mid}"

    # ── entry point ──────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load(state.traderData)
        orders: Dict[str, List[Order]] = {}

        pos_aco = state.position.get("ASH_COATED_OSMIUM", 0)
        max_vol = ACO_MAX_VOL // 2 if abs(pos_aco) > 120 else ACO_MAX_VOL

        od = state.order_depths.get("ASH_COATED_OSMIUM")
        if od is not None:
            orders["ASH_COATED_OSMIUM"] = self._aco(od, pos_aco, max_vol)

        trader_data = self._save()
        logger.flush(state, orders, 0, trader_data)
        return orders, 0, trader_data

    # ── ASH_COATED_OSMIUM — market making ────────────────────────────────────

    def _aco(self, od: OrderDepth, pos: int, max_vol: int) -> List[Order]:
        result: List[Order] = []

        asks = sorted(od.sell_orders.items())
        bids = sorted(od.buy_orders.items(), reverse=True)

        if not asks or not bids:
            return result

        ba  = asks[0][0]
        bb  = bids[0][0]
        mid = (ba + bb) / 2.0

        self.aco_ema = ACO_EMA_ALPHA * mid + (1 - ACO_EMA_ALPHA) * self.aco_ema
        fv = ACO_FV

        # TAKE buy
        for ask, ask_vol in asks:
            if ask > fv - ACO_TAKE_EDGE:
                break
            buy_cap = POSITION_LIMIT - pos
            if buy_cap <= 0:
                break
            vol = int(min(abs(ask_vol), buy_cap))
            if vol > 0:
                result.append(Order("ASH_COATED_OSMIUM", ask, vol))
                pos += vol

        # TAKE sell
        for bid, bid_vol in bids:
            if bid < fv + ACO_TAKE_EDGE:
                break
            sell_cap = POSITION_LIMIT + pos
            if sell_cap <= 0:
                break
            vol = int(min(bid_vol, sell_cap))
            if vol > 0:
                result.append(Order("ASH_COATED_OSMIUM", bid, -vol))
                pos -= vol

        # MAKE: inventory skew + momentum skew + imbalance signal
        last_ret_aco  = (mid - self.prev_mid_aco) if self.prev_mid_aco is not None else 0.0
        self.prev_mid_aco = mid
        momentum_skew = -last_ret_aco * MOMENTUM_SKEW_FACTOR

        total_bid_vol  = sum(v for _, v in bids)
        total_ask_vol  = sum(abs(v) for _, v in asks)
        total_vol      = total_bid_vol + total_ask_vol
        imbalance      = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0.0
        imbalance_skew = imbalance * IMBALANCE_FACTOR

        inventory_skew = pos * ACO_SKEW_FACTOR

        buy_px_f  = fv - ACO_SPREAD - inventory_skew + momentum_skew + imbalance_skew
        sell_px_f = fv + ACO_SPREAD - inventory_skew + momentum_skew + imbalance_skew

        buy_px_f  = min(max(buy_px_f,  bb + 1), fv - 1.0)
        sell_px_f = max(min(sell_px_f, ba - 1), fv + 1.0)

        buy_px  = round(buy_px_f)
        sell_px = round(sell_px_f)

        buy_vol  = int(min(max_vol, POSITION_LIMIT - pos))
        sell_vol = int(min(max_vol, POSITION_LIMIT + pos))

        vol1 = buy_vol // 2
        vol2 = buy_vol - vol1
        if vol1 > 0 and buy_px > 0:
            result.append(Order("ASH_COATED_OSMIUM", buy_px, vol1))
        if vol2 > 0 and buy_px - 1 > 0:
            result.append(Order("ASH_COATED_OSMIUM", buy_px - 1, vol2))

        vol1 = sell_vol // 2
        vol2 = sell_vol - vol1
        if vol1 > 0:
            result.append(Order("ASH_COATED_OSMIUM", sell_px, -vol1))
        if vol2 > 0:
            result.append(Order("ASH_COATED_OSMIUM", sell_px + 1, -vol2))

        return result
