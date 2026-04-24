import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

round3_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/ROUND_3")
output_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/Test Round 3 ")
product = 'VEV_4000'

# Laden der Daten
price_dfs = {}
for day in range(3):
    df = pd.read_csv(round3_path / f"prices_round_3_day_{day}.csv", sep=';')
    df = df[df['product'] == product].copy()
    df['day'] = pd.to_numeric(df['day'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['mid_price'] = pd.to_numeric(df['mid_price'])
    df['abs_timestamp'] = df['day'] * 10000 + df['timestamp']
    price_dfs[day] = df.sort_values('abs_timestamp').reset_index(drop=True)

if price_dfs[0].empty:
    raise ValueError(f'Day 0 Daten für Produkt {product} fehlen')

# Referenz aus Tag 0: sequential record lows und highs
ref_prices = price_dfs[0]['mid_price'].tolist()
ref_lows = []
ref_highs = []
current_low = float('inf')
current_high = float('-inf')
for p in ref_prices:
    if p < current_low:
        current_low = p
        ref_lows.append(p)
    if p > current_high:
        current_high = p
        ref_highs.append(p)

print('Day 0 reference lows:', ref_lows)
print('Day 0 reference highs:', ref_highs)


def simulate_day(day_df):
    trades = []
    hold_prices = []
    buy_index = 0
    last_sell_level_idx = 0

    def get_sell_level():
        if not hold_prices:
            return None
        current_max_buy = max(hold_prices)
        for idx, level in enumerate(ref_highs):
            if level > current_max_buy:
                return idx, level
        return None, None

    for _, row in day_df.iterrows():
        price = row['mid_price']
        ts = row['abs_timestamp']

        # Sell if held and price reaches next sell level
        if hold_prices:
            sell_idx, sell_level = get_sell_level()
            if sell_level is not None and price >= sell_level:
                quantity = len(hold_prices)
                avg_buy = sum(hold_prices) / quantity
                pnl = (price - avg_buy) * quantity
                trades.append({
                    'action': 'SELL',
                    'price': price,
                    'quantity': quantity,
                    'timestamp': ts,
                    'avg_buy': avg_buy,
                    'pnl': pnl,
                    'sell_level': sell_level,
                })
                hold_prices = []
                buy_index = 0
                last_sell_level_idx = 0
                continue

        # Buy if price reaches the next lower bound
        if buy_index < len(ref_lows) and price <= ref_lows[buy_index]:
            trades.append({
                'action': 'BUY',
                'price': price,
                'quantity': 1,
                'timestamp': ts,
                'buy_level': ref_lows[buy_index],
            })
            hold_prices.append(price)
            buy_index += 1
            continue

    # Close any remaining position at last price of the day
    if hold_prices:
        last_price = day_df.iloc[-1]['mid_price']
        quantity = len(hold_prices)
        avg_buy = sum(hold_prices) / quantity
        pnl = (last_price - avg_buy) * quantity
        trades.append({
            'action': 'CLOSE',
            'price': last_price,
            'quantity': quantity,
            'timestamp': day_df.iloc[-1]['abs_timestamp'],
            'avg_buy': avg_buy,
            'pnl': pnl,
            'note': 'end_of_day_close',
        })
    return trades

results = {}
for day in [1, 2]:
    trades = simulate_day(price_dfs[day])
    total_pnl = sum(t['pnl'] for t in trades if t['action'] in ('SELL', 'CLOSE'))
    total_qty = sum(t['quantity'] for t in trades if t['action'] == 'BUY')
    results[day] = {
        'trades': trades,
        'pnl': total_pnl,
        'total_buys': total_qty,
        'total_sells': sum(1 for t in trades if t['action'] == 'SELL'),
        'end_closes': sum(1 for t in trades if t['action'] == 'CLOSE'),
    }

# Output summary
for day in [1, 2]:
    print(f"\n--- Day {day} ---")
    print(f"Buys: {results[day]['total_buys']}")
    print(f"Sells: {results[day]['total_sells']}")
    print(f"End of day closes: {results[day]['end_closes']}")
    print(f"Total PnL: {results[day]['pnl']:.2f}")
    for trade in results[day]['trades']:
        if trade['action'] == 'BUY':
            print(f"BUY  @ {trade['price']:.2f} (level {trade['buy_level']:.2f}) at {trade['timestamp']}")
        elif trade['action'] == 'SELL':
            print(f"SELL @ {trade['price']:.2f} (level {trade['sell_level']:.2f}) qty {trade['quantity']} pnl {trade['pnl']:.2f}")
        else:
            print(f"CLOSE @ {trade['price']:.2f} qty {trade['quantity']} pnl {trade['pnl']:.2f}")

# Plot day 1 and day 2 with buy/sell markers
plt.figure(figsize=(14, 7))
for day in [1, 2]:
    df = price_dfs[day]
    color = 'blue' if day == 1 else 'orange'
    plt.plot(df['abs_timestamp'], df['mid_price'], marker='o', markersize=4, linestyle='-', label=f'Day {day}')
    for trade in results[day]['trades']:
        if trade['action'] == 'BUY':
            plt.scatter(trade['timestamp'], trade['price'], color='green', s=100, marker='^')
        elif trade['action'] == 'SELL':
            plt.scatter(trade['timestamp'], trade['price'], color='red', s=100, marker='v')
        elif trade['action'] == 'CLOSE':
            plt.scatter(trade['timestamp'], trade['price'], color='purple', s=100, marker='X')

for level in ref_lows:
    plt.axhline(level, color='gray', linestyle='--', alpha=0.15)
for level in ref_highs:
    plt.axhline(level, color='red', linestyle=':', alpha=0.2)

plt.title(f'Day 0 Reference Strategy for {product}')
plt.xlabel('Abs Timestamp')
plt.ylabel('Mid Price')
plt.legend()
plt.tight_layout()
plt.savefig(output_path / 'day0_ref_strategy.png', dpi=300)
print(f"\nGrafik gespeichert: {output_path / 'day0_ref_strategy.png'}")
