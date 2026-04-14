"""
trader_r1.py — IMC Prosperity Round 1 (Intara)
───────────────────────────────────────────────────────────────────
INTARIAN_PEPPER_ROOT: FV = 10000 + 1000*day + 0.001*timestamp (linear drift)
ASH_COATED_OSMIUM:   FV = 10000 fix, light EMA (alpha=0.05) pentru take edge

Optimizări v4:
  - cache local sell_map/buy_map (evită attribute lookup repetat)
  - iterare items() în TAKE (evită double dict lookup)
  - clamp inlinat ca int(min(...)) (elimină function call overhead)
  - dispatch direct pe produs (fără loop cu string compare)
  - momentum skew mean-reversion pe ambele produse
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional

# ── limite ────────────────────────────────────────────────────────────────────

POSITION_LIMITS: Dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# INTARIAN_PEPPER_ROOT
IPR_SPREAD      = 5
IPR_MAX_VOL     = 25
IPR_TAKE_EDGE   = 2
IPR_SKEW_FACTOR = 0.05

# ASH_COATED_OSMIUM
ACO_FV          = 10_000
ACO_EMA_ALPHA   = 0.05
ACO_SPREAD      = 6
ACO_MAX_VOL     = 20
ACO_TAKE_EDGE   = 4
ACO_SKEW_FACTOR = 0.10

MOMENTUM_SKEW_FACTOR = 0.3  # mean reversion coefficient, empirically derived


class Trader:

    def __init__(self):
        self.current_day: int          = 0
        self.prev_ts: int              = 0
        self.aco_ema: float            = 10_000.0
        self.prev_mid_ipr: Optional[float] = None
        self.prev_mid_aco: Optional[float] = None

    # ── persistență pipe-separated ───────────────────────────────────────────
    # format: "current_day|prev_timestamp|aco_ema|prev_mid_ipr|prev_mid_aco"

    def _load(self, raw: str, timestamp: int):
        if not raw:
            self.current_day  = 0
            self.prev_ts      = timestamp
            self.aco_ema      = 10_000.0
            self.prev_mid_ipr = None
            self.prev_mid_aco = None
            return
        parts            = raw.split("|")
        self.current_day = int(parts[0])
        self.prev_ts     = int(parts[1])
        self.aco_ema     = float(parts[2])
        self.prev_mid_ipr = float(parts[3]) if len(parts) > 3 and parts[3] != "None" else None
        self.prev_mid_aco = float(parts[4]) if len(parts) > 4 and parts[4] != "None" else None
        if timestamp < self.prev_ts:
            self.current_day += 1
        self.prev_ts = timestamp

    def _save(self) -> str:
        ipr_mid = "None" if self.prev_mid_ipr is None else str(self.prev_mid_ipr)
        aco_mid = "None" if self.prev_mid_aco is None else str(self.prev_mid_aco)
        return f"{self.current_day}|{self.prev_ts}|{self.aco_ema}|{ipr_mid}|{aco_mid}"

    # ── entry point ──────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load(state.traderData, state.timestamp)
        orders: Dict[str, List[Order]] = {}

        pos_ipr = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        pos_aco = state.position.get("ASH_COATED_OSMIUM", 0)
        total_inv = abs(pos_ipr) + abs(pos_aco)

        ipr_max_vol = IPR_MAX_VOL // 2 if total_inv > 120 else IPR_MAX_VOL
        aco_max_vol = ACO_MAX_VOL // 2 if total_inv > 120 else ACO_MAX_VOL

        od = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if od is not None:
            orders["INTARIAN_PEPPER_ROOT"] = self._ipr(od, pos_ipr, 80, state.timestamp, ipr_max_vol)

        od = state.order_depths.get("ASH_COATED_OSMIUM")
        if od is not None:
            orders["ASH_COATED_OSMIUM"] = self._aco(od, pos_aco, 80, aco_max_vol)

        return orders, 0, self._save()

    # ── INTARIAN_PEPPER_ROOT ──────────────────────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, limit: int, timestamp: int, max_vol: int) -> List[Order]:
        result: List[Order] = []
        fv = 12_000 + 1_000 * self.current_day + 0.001 * timestamp

        sell_map = od.sell_orders
        buy_map  = od.buy_orders

        asks = sorted(sell_map.items())              # (price, neg_vol) ascending
        bids = sorted(buy_map.items(), reverse=True) # (price, vol) descending

        # TAKE buy
        for ask, ask_vol in asks:
            if ask > fv - IPR_TAKE_EDGE:
                break
            buy_cap = limit - pos
            if buy_cap <= 0:
                break
            vol = int(min(abs(ask_vol), buy_cap))
            if vol > 0:
                result.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                pos += vol

        # TAKE sell
        for bid, bid_vol in bids:
            if bid < fv + IPR_TAKE_EDGE:
                break
            sell_cap = limit + pos
            if sell_cap <= 0:
                break
            vol = int(min(bid_vol, sell_cap))
            if vol > 0:
                result.append(Order("INTARIAN_PEPPER_ROOT", bid, -vol))
                pos -= vol

        # MAKE cu inventory skew + momentum skew
        bb = bids[0][0] if bids else None
        ba = asks[0][0] if asks else None

        if bb is not None and ba is not None:
            mid = (ba + bb) / 2.0
            last_ret_ipr = (mid - self.prev_mid_ipr) if self.prev_mid_ipr is not None else 0.0
            self.prev_mid_ipr = mid
        else:
            last_ret_ipr = 0.0

        momentum_skew_ipr = -last_ret_ipr * MOMENTUM_SKEW_FACTOR
        inventory_skew    = pos * IPR_SKEW_FACTOR

        buy_px_f  = fv - IPR_SPREAD - inventory_skew + momentum_skew_ipr
        sell_px_f = fv + IPR_SPREAD - inventory_skew + momentum_skew_ipr

        buy_px_f  = min(max(buy_px_f,  (bb + 1) if bb is not None else buy_px_f),  fv - 1.0)
        sell_px_f = max(min(sell_px_f, (ba - 1) if ba is not None else sell_px_f), fv + 1.0)

        buy_px  = round(buy_px_f)
        sell_px = round(sell_px_f)

        buy_vol  = int(min(max_vol, limit - pos))
        sell_vol = int(min(max_vol, limit + pos))

        if buy_vol > 0:
            result.append(Order("INTARIAN_PEPPER_ROOT", buy_px, buy_vol))
        if sell_vol > 0:
            result.append(Order("INTARIAN_PEPPER_ROOT", sell_px, -sell_vol))

        return result

    # ── ASH_COATED_OSMIUM ─────────────────────────────────────────────────────

    def _aco(self, od: OrderDepth, pos: int, limit: int, max_vol: int) -> List[Order]:
        result: List[Order] = []

        sell_map = od.sell_orders
        buy_map  = od.buy_orders

        asks = sorted(sell_map.items())
        bids = sorted(buy_map.items(), reverse=True)

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
            buy_cap = limit - pos
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
            sell_cap = limit + pos
            if sell_cap <= 0:
                break
            vol = int(min(bid_vol, sell_cap))
            if vol > 0:
                result.append(Order("ASH_COATED_OSMIUM", bid, -vol))
                pos -= vol

        # MAKE cu inventory skew + momentum skew
        last_ret_aco = (mid - self.prev_mid_aco) if self.prev_mid_aco is not None else 0.0
        self.prev_mid_aco = mid
        momentum_skew_aco = -last_ret_aco * MOMENTUM_SKEW_FACTOR
        inventory_skew    = pos * ACO_SKEW_FACTOR

        buy_px_f  = ACO_FV - ACO_SPREAD - inventory_skew + momentum_skew_aco
        sell_px_f = ACO_FV + ACO_SPREAD - inventory_skew + momentum_skew_aco

        buy_px_f  = min(max(buy_px_f,  bb + 1), fv - 1.0)
        sell_px_f = max(min(sell_px_f, ba - 1), fv + 1.0)

        buy_px  = round(buy_px_f)
        sell_px = round(sell_px_f)

        buy_vol  = int(min(max_vol, limit - pos))
        sell_vol = int(min(max_vol, limit + pos))

        if buy_vol > 0 and buy_px > 0:
            result.append(Order("ASH_COATED_OSMIUM", buy_px, buy_vol))
        if sell_vol > 0:
            result.append(Order("ASH_COATED_OSMIUM", sell_px, -sell_vol))

        return result
