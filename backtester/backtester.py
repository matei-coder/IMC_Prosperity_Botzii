import pandas as pd
from typing import Dict, List
from datamodel import TradingState, OrderDepth, Order
import matplotlib.pyplot as plt
from trader import Trader

class Backtester:
    def __init__(self, trader, csv_file_path):
        self.last_mid_prices: Dict[str, float] = {}
        self.trader = trader
        self.df = pd.read_csv(csv_file_path, sep=';')
        self.positions: Dict[str, int] = {}
        self.cash: float = 0.0
        self.pnl_history: List[float] = []
        self.timestamps_history: List[int] = []

    def build_order_depths(self, row) -> OrderDepth:
        order_depth = OrderDepth()

        for i in range(1, 4):
            price_col = f'bid_price_{i}'
            vol_col = f'bid_volume_{i}'

            if pd.notna(row[price_col]) and pd.notna(row[vol_col]):
                price = int(row[price_col])
                volume = int(row[vol_col])
                order_depth.buy_orders[price] = volume

        for i in range(1, 4):
            price_col = f'ask_price_{i}'
            vol_col = f'ask_volume_{i}'

            if pd.notna(row[price_col]) and pd.notna(row[vol_col]):
                price = int(row[price_col])
                volume = int(row[vol_col])
                order_depth.sell_orders[price] = -volume

        return order_depth

    def simulate_matching(self, symbol: str, orders: List[Order], order_depth: OrderDepth):

        if symbol not in self.positions:
            self.positions[symbol] = 0

        for order in orders:

            if order.quantity > 0:
                available_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])

                for ask_price, ask_vol in available_asks:
                    if order.quantity == 0:
                        break

                    if ask_price <= order.price:
                        available_vol = abs(ask_vol)
                        trade_vol = min(order.quantity, available_vol)

                        self.positions[symbol] += trade_vol
                        self.cash -= trade_vol * ask_price

                        order.quantity -= trade_vol
                        order_depth.sell_orders[
                            ask_price] += trade_vol

            elif order.quantity < 0:
                available_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

                qty_to_sell = abs(order.quantity)

                for bid_price, bid_vol in available_bids:
                    if qty_to_sell == 0:
                        break

                    if bid_price >= order.price:
                        trade_vol = min(qty_to_sell, bid_vol)

                        self.positions[symbol] -= trade_vol
                        self.cash += trade_vol * bid_price

                        qty_to_sell -= trade_vol
                        order.quantity += trade_vol
                        order_depth.buy_orders[bid_price] -= trade_vol

    def calculate_pnl(self, current_order_depths: Dict[str, OrderDepth]) -> float:
        total_pnl = self.cash

        for symbol, pos in self.positions.items():
            if pos == 0:
                continue

            depth = current_order_depths.get(symbol)

            if depth and (depth.buy_orders or depth.sell_orders):
                best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
                best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0

                if best_bid > 0 and best_ask > 0:
                    mid_price = (best_bid + best_ask) / 2.0
                elif best_bid > 0:
                    mid_price = best_bid
                elif best_ask > 0:
                    mid_price = best_ask
                else:
                    mid_price = 0

                self.last_mid_prices[symbol] = mid_price
            else:
                mid_price = self.last_mid_prices.get(symbol, 0)

            total_pnl += pos * mid_price

        return total_pnl

    def run(self):
        timestamps = sorted(self.df['timestamp'].unique())

        for ts in timestamps:
            rows = self.df[self.df['timestamp'] == ts]

            current_order_depths = {}
            for _, row in rows.iterrows():
                product = row['product']
                current_order_depths[product] = self.build_order_depths(row)

            state = TradingState(
                traderData="",
                timestamp=ts,
                listings={},
                order_depths=current_order_depths,
                own_trades={},
                market_trades={},
                position=self.positions.copy(),
                observations={}
            )
            orders_dict, conversions, traderData = self.trader.run(state)

            for symbol, orders in orders_dict.items():
                if symbol in current_order_depths:
                    self.simulate_matching(symbol, orders, current_order_depths[symbol])

            current_pnl = self.calculate_pnl(current_order_depths)
            self.pnl_history.append(current_pnl)
            self.timestamps_history.append(ts)

        print("Backtest Finalizat.")
        if len(self.pnl_history) > 0:
            print(f"PnL Final: {self.pnl_history[-1]:.2f}")

    def plot_results(self):

        plt.figure(figsize=(12, 6))

        plt.plot(self.timestamps_history, self.pnl_history, label="Total PnL", color='#1f77b4', linewidth=2)

        plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        plt.title("Evoluția Profit & Loss", fontsize=14, fontweight='bold')
        plt.xlabel("Timestamp", fontsize=12)
        plt.ylabel("PnL", fontsize=12)

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    my_trader = Trader()
    tester = Backtester(my_trader, 'prices_round_1_day_0.csv')
    tester.run()
    tester.plot_results()