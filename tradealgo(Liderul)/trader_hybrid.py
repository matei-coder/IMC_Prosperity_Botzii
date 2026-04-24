from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional

# ── constante ─────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
SELL_TS_START  = 80_000   # IPR: buffer de 20k pentru a evita deșertul de lichiditate

ACO_FV               = 10_000
ACO_EMA_ALPHA        = 0.11
ACO_SPREAD           = 18
ACO_MAX_VOL          = 20
ACO_TAKE_EDGE        = 4
ACO_SKEW_FACTOR      = 0.10
MOMENTUM_SKEW_FACTOR = 0.3
IMBALANCE_FACTOR     = 1.5

# IPR Specific
IPR_EMA_ALPHA        = 0.15
IPR_SELL_FLOOR_BASE  = 12_000
IPR_TAKE_SPREAD_MAX  = 5  # FIX: acumulare in transe cu limita de spread

class Trader:

    def __init__(self):
        self.current_day: int              = 0
        self.prev_ts: int                  = 0
        self.aco_ema: float                = 10_000.0
        self.ipr_ema: float                = 12_000.0
        self.prev_mid_aco: Optional[float] = None

    # ── persistență pipe-separated ───────────────────────────────────────────

    def _load(self, raw: str, timestamp: int):
        if not raw:
            self.current_day  = 0
            self.prev_ts      = timestamp
            self.aco_ema      = 10_000.0
            self.ipr_ema      = 12_000.0
            self.prev_mid_aco = None
            return
        
        parts = raw.split("|")
        self.current_day  = int(parts[0])
        self.prev_ts      = int(parts[1])
        self.aco_ema      = float(parts[2])
        
        # Backward compatibility pentru cand ipr_ema a fost adaugat
        if len(parts) > 3:
            self.ipr_ema = float(parts[3])
        else:
            self.ipr_ema = 12_000.0
            
        if len(parts) > 4 and parts[4] != "None":
            self.prev_mid_aco = float(parts[4])
        else:
            self.prev_mid_aco = None
            
        if timestamp < self.prev_ts:
            self.current_day += 1
        self.prev_ts = timestamp

    def _save(self) -> str:
        aco_mid = "None" if self.prev_mid_aco is None else str(self.prev_mid_aco)
        return f"{self.current_day}|{self.prev_ts}|{self.aco_ema}|{self.ipr_ema}|{aco_mid}"

    # ── entry point ──────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load(state.traderData, state.timestamp)
        orders: Dict[str, List[Order]] = {}

        pos_ipr = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        pos_aco = state.position.get("ASH_COATED_OSMIUM", 0)

        od_ipr = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if od_ipr is not None:
            orders["INTARIAN_PEPPER_ROOT"] = self._ipr(od_ipr, pos_ipr, state.timestamp)

        od_aco = state.order_depths.get("ASH_COATED_OSMIUM")
        if od_aco is not None:
            aco_max_vol = ACO_MAX_VOL // 2 if abs(pos_aco) > 120 else ACO_MAX_VOL
            orders["ASH_COATED_OSMIUM"] = self._aco(od_aco, pos_aco, aco_max_vol)

        return orders, 0, self._save()

    # ── INTARIAN_PEPPER_ROOT — directional ───────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        result: List[Order] = []

        bids = sorted(od.buy_orders.items(), reverse=True)
        asks = sorted(od.sell_orders.items())

        # Update sigur la EMA
        if bids and asks:  # FIX 3
            ipr_mid = (asks[0][0] + bids[0][0]) / 2.0
            self.ipr_ema = 0.05 * ipr_mid + 0.95 * self.ipr_ema

        if timestamp >= SELL_TS_START:
            # Sell phase: Hibrid EMA + Floor
            sell_floor = IPR_SELL_FLOOR_BASE + 1_000 * self.current_day  # FIX 3
            fv_ipr = max(self.ipr_ema, sell_floor)  # FIX 3

            # 1. Take agresiv: hit bid-uri
            for bid, bid_vol in bids:
                if pos <= 0:
                    break
                vol = int(min(bid_vol, pos))
                if vol > 0:
                    # Ne asiguram ca nu vindem sub pragul pieței valabile
                    limit_sell = max(bid, round(fv_ipr) - 2)
                    result.append(Order("INTARIAN_PEPPER_ROOT", limit_sell, -vol))
                    pos -= vol

            # 2. Make pasiv: plasam ordine pentru restul portofoliului  # FIX 2
            if pos > 0 and bids:
                passive_sell_px = bids[0][0] + 1  # un tick peste best bid
                passive_sell_px = max(passive_sell_px, sell_floor)
                result.append(Order("INTARIAN_PEPPER_ROOT", passive_sell_px, -pos))
        else:
            # Accumulation phase
            for ask, ask_vol in asks:
                buy_cap = POSITION_LIMIT - pos
                if buy_cap <= 0:
                    break
                if ask - ipr_mid > IPR_TAKE_SPREAD_MAX:  # FIX: acumulare in transe cu limita de spread
                    break  # spread prea mare, nu lifta agresiv
                vol = int(min(abs(ask_vol), buy_cap))
                if vol > 0:
                    result.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                    pos += vol

            # bid pasiv pentru capacitatea rămasă
            if pos < POSITION_LIMIT:
                buy_cap = POSITION_LIMIT - pos
                passive_px = round(ipr_mid) - 1
                if asks:
                    passive_px = min(passive_px, asks[0][0] - 1)
                result.append(Order("INTARIAN_PEPPER_ROOT", passive_px, buy_cap))

        return result

    # ── ASH_COATED_OSMIUM — market making ────────────────────────────────────

    def _aco(self, od: OrderDepth, pos: int, max_vol: int) -> List[Order]:
        result: List[Order] = []

        asks = sorted(od.sell_orders.items())
        bids = sorted(od.buy_orders.items(), reverse=True)

        # Fix extrem pentru Flash Crash & Zero Value Bugs
        if not asks or not bids:
            return result
        
        ba = asks[0][0]
        bb = bids[0][0]

        if ba == 0 or bb == 0 or ba <= bb:  # FIX 1
            return result

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
        if mid == 0 or self.prev_mid_aco is None or self.prev_mid_aco == 0:  # FIX 1
            momentum_skew = 0.0
        else:
            momentum_skew = -(mid - self.prev_mid_aco) * MOMENTUM_SKEW_FACTOR
        self.prev_mid_aco = mid

        total_bid_vol  = sum(v for _, v in bids)
        total_ask_vol  = sum(abs(v) for _, v in asks)
        total_vol      = total_bid_vol + total_ask_vol
        imbalance      = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0.0
        imbalance_skew = imbalance * IMBALANCE_FACTOR

        inventory_skew = pos * ACO_SKEW_FACTOR

        buy_px_f  = fv - ACO_SPREAD - inventory_skew + momentum_skew + imbalance_skew
        sell_px_f = fv + ACO_SPREAD - inventory_skew + momentum_skew + imbalance_skew

        # Protecții stricte de limite pentru a sta aproape de book și FV
        buy_px_f  = min(max(buy_px_f,  bb + 1), fv - 1.0)
        sell_px_f = max(min(sell_px_f, ba - 1), fv + 1.0)

        buy_px  = round(buy_px_f)
        sell_px = round(sell_px_f)

        # Plasarea limitelor finale - Cumpărare
        buy_vol  = int(min(max_vol, POSITION_LIMIT - pos))
        if buy_vol > 0 and buy_px > 0:
            vol1 = buy_vol // 2
            vol2 = buy_vol - vol1
            if vol1 > 0:
                result.append(Order("ASH_COATED_OSMIUM", buy_px, vol1))
            if vol2 > 0 and buy_px - 1 > 0:
                result.append(Order("ASH_COATED_OSMIUM", buy_px - 1, vol2))

        # Plasarea limitelor finale - Vânzare
        sell_vol = int(min(max_vol, POSITION_LIMIT + pos))
        if sell_vol > 0 and sell_px > 0:
            vol1 = sell_vol // 2
            vol2 = sell_vol - vol1
            if vol1 > 0:
                result.append(Order("ASH_COATED_OSMIUM", sell_px, -vol1))
            if vol2 > 0:
                result.append(Order("ASH_COATED_OSMIUM", sell_px + 1, -vol2))

        return result