from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional

# ── constante ─────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
# Buffer mărit: începem vânzarea la 80k pentru a evita deșertul de lichiditate
SELL_TS_START  = 80_000   

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

class Trader:

    def __init__(self):
        self.current_day: int              = 0
        self.prev_ts: int                  = 0
        self.aco_ema: float                = 10_000.0
        self.ipr_ema: float                = 12_000.0 # Valoare inițială estimată
        self.prev_mid_aco: Optional[float] = None

    # ── persistență pipe-separated ───────────────────────────────────────────
    # format: "current_day|prev_timestamp|aco_ema|ipr_ema|prev_mid_aco"

    def _load(self, raw: str, timestamp: int):
        if not raw:
            return
        parts             = raw.split("|")
        self.current_day  = int(parts[0])
        self.prev_ts      = int(parts[1])
        self.aco_ema      = float(parts[2])
        self.ipr_ema      = float(parts[3]) if len(parts) > 3 else 12_000.0
        self.prev_mid_aco = float(parts[4]) if len(parts) > 4 and parts[4] != "None" else None
        
        if timestamp < self.prev_ts:
            self.current_day += 1
        self.prev_ts = timestamp

    def _save(self) -> str:
        aco_mid = "None" if self.prev_mid_aco is None else str(self.prev_mid_aco)
        return f"{self.current_day}|{self.prev_ts}|{self.aco_ema}|{self.ipr_ema}|{aco_mid}"

    def run(self, state: TradingState):
        self._load(state.traderData, state.timestamp)
        orders: Dict[str, List[Order]] = {}

        pos_ipr = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        pos_aco = state.position.get("ASH_COATED_OSMIUM", 0)

        # ── Update IPR EMA ──
        od_ipr = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if od_ipr and od_ipr.buy_orders and od_ipr.sell_orders:
            mid_ipr = (max(od_ipr.buy_orders.keys()) + min(od_ipr.sell_orders.keys())) / 2.0
            self.ipr_ema = IPR_EMA_ALPHA * mid_ipr + (1 - IPR_EMA_ALPHA) * self.ipr_ema
            orders["INTARIAN_PEPPER_ROOT"] = self._ipr(od_ipr, pos_ipr, state.timestamp)

        # ── Update ACO ──
        od_aco = state.order_depths.get("ASH_COATED_OSMIUM")
        if od_aco:
            aco_max_vol = ACO_MAX_VOL // 2 if abs(pos_aco) > 120 else ACO_MAX_VOL
            orders["ASH_COATED_OSMIUM"] = self._aco(od_aco, pos_aco, aco_max_vol)

        return orders, 0, self._save()

    # ── INTARIAN_PEPPER_ROOT ─────────────────────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        result: List[Order] = []
        bids = sorted(od.buy_orders.items(), reverse=True)
        asks = sorted(od.sell_orders.items())

        if timestamp >= SELL_TS_START:
            # Sell phase: Take agresiv + Passive sell orders
            floor = IPR_SELL_FLOOR_BASE + (1_000 * self.current_day)
            fv_sell = max(self.ipr_ema, floor)

            # 1. Take bids
            for bid, bid_vol in bids:
                if pos <= 0: break
                vol = int(min(bid_vol, pos))
                if vol > 0:
                    # Nu vindem sub floor-ul de profitabilitate decât dacă e critic
                    price = max(bid, round(fv_sell) - 1)
                    result.append(Order("INTARIAN_PEPPER_ROOT", price, -vol))
                    pos -= vol
            
            # 2. Make sell (ordine pasive pentru restul de poziție)
            if pos > 0:
                # Plasăm ordin la un tick deasupra celui mai bun bid sau la FV
                sell_px = max(bids[0][0] + 1, round(fv_sell)) if bids else round(fv_sell)
                result.append(Order("INTARIAN_PEPPER_ROOT", sell_px, -pos))
        else:
            # Accumulation phase
            for ask, ask_vol in asks:
                buy_cap = POSITION_LIMIT - pos
                if buy_cap <= 0: break
                vol = int(min(abs(ask_vol), buy_cap))
                if vol > 0:
                    result.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                    pos += vol
            if pos < POSITION_LIMIT and asks:
                result.append(Order("INTARIAN_PEPPER_ROOT", asks[0][0] - 1, POSITION_LIMIT - pos))

        return result

    # ── ASH_COATED_OSMIUM ────────────────────────────────────────────────────

    def _aco(self, od: OrderDepth, pos: int, max_vol: int) -> List[Order]:
        result: List[Order] = []
        asks = sorted(od.sell_orders.items())
        bids = sorted(od.buy_orders.items(), reverse=True)

        # Fix Flash Crash: Validăm existența prețurilor și prețuri non-zero
        if not asks or not bids or asks[0][0] <= 0 or bids[0][0] <= 0:
            return result

        ba, bb = asks[0][0], bids[0][0]
        mid = (ba + bb) / 2.0

        # Momentum protection: dacă mid-ul scade aberant, ignorăm saltul
        if self.prev_mid_aco is not None and abs(mid - self.prev_mid_aco) > 500:
            last_ret_aco = 0.0
        else:
            last_ret_aco = (mid - self.prev_mid_aco) if self.prev_mid_aco is not None else 0.0
        
        self.prev_mid_aco = mid
        self.aco_ema = ACO_EMA_ALPHA * mid + (1 - ACO_EMA_ALPHA) * self.aco_ema
        
        fv = ACO_FV
        # (Logica de Take/Make rămâne neschimbată, folosind noile protecții de skew)
        # [...] 
        
        # Make Skew Calculation
        momentum_skew = -last_ret_aco * MOMENTUM_SKEW_FACTOR
        inventory_skew = pos * ACO_SKEW_FACTOR
        
        # Imbalance
        total_bid_vol = sum(v for _, v in bids)
        total_ask_vol = sum(abs(v) for _, v in asks)
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        imbalance_skew = imbalance * IMBALANCE_FACTOR

        buy_px_f = fv - ACO_SPREAD - inventory_skew + momentum_skew + imbalance_skew
        sell_px_f = fv + ACO_SPREAD - inventory_skew + momentum_skew + imbalance_skew

        # Bound prețuri să nu iasă din spread sau să devină negative
        buy_px = round(min(max(buy_px_f, bb + 1), fv - 1))
        sell_px = round(max(min(sell_px_f, ba - 1), fv + 1))

        if buy_px > 0:
            result.append(Order("ASH_COATED_OSMIUM", buy_px, max_vol // 2))
            result.append(Order("ASH_COATED_OSMIUM", buy_px - 1, max_vol - (max_vol // 2)))
        
        result.append(Order("ASH_COATED_OSMIUM", sell_px, -(max_vol // 2)))
        result.append(Order("ASH_COATED_OSMIUM", sell_px + 1, -(max_vol - (max_vol // 2))))

        return result