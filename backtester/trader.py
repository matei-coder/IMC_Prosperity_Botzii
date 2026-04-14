"""
trader_hybrid.py — IMC Prosperity Round 1 (Intara)
───────────────────────────────────────────────────────────────────
INTARIAN_PEPPER_ROOT: directional — acumulare max long, sell phase la ts >= 90000
ASH_COATED_OSMIUM:   market making complet (EMA, momentum skew, imbalance, two-level)

FIXES aplicate:
  1. IPR sell phase: limit_sell = bid direct (fără FV cu timestamp care bloca execuția)
  2. ACO FV: folosește EMA în loc de anchor static 10_000
  3. ACO take edge: redus de la 4 la 2 pentru take-uri mai frecvente
  4. aco_max_vol threshold: corectat de la 120 la 60 (în range-ul POSITION_LIMIT=80)
  5. Guard anti-inversare spread pe ACO make orders
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional

# ── constante ─────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
SELL_TS_START  = 90_000   # IPR: switch la sell phase

ACO_FV               = 10_000   # anchor de referință pentru inițializarea EMA
ACO_EMA_ALPHA        = 0.10
ACO_SPREAD           = 6
ACO_MAX_VOL          = 20
ACO_TAKE_EDGE        = 2        # FIX #3: redus de la 4 la 2
ACO_SKEW_FACTOR      = 0.10
MOMENTUM_SKEW_FACTOR = 0.3
IMBALANCE_FACTOR     = 1.5

# Preț minim acceptat în sell phase IPR (protecție de floor)
IPR_SELL_FLOOR_BASE  = 11_800   # 12_000 - 200 buffer; ajustat per day în cod


class Trader:

    def __init__(self):
        self.current_day: int              = 0
        self.prev_ts: int                  = 0
        self.aco_ema: float                = float(ACO_FV)
        self.prev_mid_aco: Optional[float] = None

    # ── persistență pipe-separated ───────────────────────────────────────────
    # format: "current_day|prev_timestamp|aco_ema|prev_mid_aco"

    def _load(self, raw: str, timestamp: int):
        if not raw:
            self.current_day  = 0
            self.prev_ts      = timestamp
            self.aco_ema      = float(ACO_FV)
            self.prev_mid_aco = None
            return
        parts             = raw.split("|")
        self.current_day  = int(parts[0])
        self.prev_ts      = int(parts[1])
        self.aco_ema      = float(parts[2])
        self.prev_mid_aco = float(parts[3]) if len(parts) > 3 and parts[3] != "None" else None
        if timestamp < self.prev_ts:
            self.current_day += 1
        self.prev_ts = timestamp

    def _save(self) -> str:
        aco_mid = "None" if self.prev_mid_aco is None else str(self.prev_mid_aco)
        return f"{self.current_day}|{self.prev_ts}|{self.aco_ema}|{aco_mid}"

    # ── entry point ──────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load(state.traderData, state.timestamp)
        orders: Dict[str, List[Order]] = {}

        pos_ipr = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        pos_aco = state.position.get("ASH_COATED_OSMIUM", 0)

        # FIX #4: threshold corectat la 60 (era 120, imposibil cu POSITION_LIMIT=80)
        aco_max_vol = ACO_MAX_VOL // 2 if abs(pos_aco) > 60 else ACO_MAX_VOL

        od = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if od is not None:
            orders["INTARIAN_PEPPER_ROOT"] = self._ipr(od, pos_ipr, state.timestamp)

        od = state.order_depths.get("ASH_COATED_OSMIUM")
        if od is not None:
            orders["ASH_COATED_OSMIUM"] = self._aco(od, pos_aco, aco_max_vol)

        return orders, 0, self._save()

    # ── INTARIAN_PEPPER_ROOT — directional ───────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        result: List[Order] = []

        bids = sorted(od.buy_orders.items(), reverse=True)
        asks = sorted(od.sell_orders.items())

        if timestamp >= SELL_TS_START:
            # FIX #1: sell phase agresivă — vinde direct la bid fără FV cu timestamp
            # Floor de protecție bazat pe day, fără componenta 0.001*ts care bloca execuția
            sell_floor = IPR_SELL_FLOOR_BASE + 1_000 * self.current_day
            for bid, bid_vol in bids:
                sell_cap = pos
                if sell_cap <= 0:
                    break
                # Refuză numai dacă bid-ul e sub floor-ul de protecție
                if bid < sell_floor:
                    break
                vol = int(min(bid_vol, sell_cap))
                if vol > 0:
                    result.append(Order("INTARIAN_PEPPER_ROOT", bid, -vol))
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
            # passive bid pentru capacitatea rămasă
            if pos < POSITION_LIMIT and asks:
                buy_cap = POSITION_LIMIT - pos
                result.append(Order("INTARIAN_PEPPER_ROOT", asks[0][0] - 1, buy_cap))

        return result

    # ── ASH_COATED_OSMIUM — market making ────────────────────────────────────

    def _aco(self, od: OrderDepth, pos: int, max_vol: int) -> List[Order]:
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

        # FIX #2: actualizează EMA și folosește-o ca FV în loc de anchor static
        self.aco_ema = ACO_EMA_ALPHA * mid + (1 - ACO_EMA_ALPHA) * self.aco_ema
        fv = self.aco_ema  # era: fv = ACO_FV (constant 10_000, EMA ignorată)

        # TAKE buy: cumperi dacă ask e suficient de mic față de FV
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

        # TAKE sell: vinzi dacă bid e suficient de mare față de FV
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

        # FIX #5: guard anti-inversare spread — dacă skew-urile au inversat prețurile
        if buy_px >= sell_px:
            buy_px  = round(fv) - 1
            sell_px = round(fv) + 1

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