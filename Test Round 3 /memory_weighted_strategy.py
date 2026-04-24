import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

round3_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/ROUND_3")
output_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/Test Round 3 ")
product = 'VEV_4000'

# Hilfsfunktionen

def detect_local_levels(prices):
    lows = []
    highs = []
    n = len(prices)
    if n == 0:
        return lows, highs
    for i in range(n):
        p = prices[i]
        prev_p = prices[i - 1] if i > 0 else None
        next_p = prices[i + 1] if i < n - 1 else None
        if prev_p is not None and next_p is not None:
            if p <= prev_p and p <= next_p:
                lows.append(p)
            if p >= prev_p and p >= next_p:
                highs.append(p)
        elif prev_p is None and next_p is not None:
            if p <= next_p:
                lows.append(p)
            if p >= next_p:
                highs.append(p)
        elif next_p is None and prev_p is not None:
            if p <= prev_p:
                lows.append(p)
            if p >= prev_p:
                highs.append(p)
    return lows, highs


def merge_level(levels, price, level_type, weight, tolerance=1.0):
    closest = None
    closest_dist = None
    for lvl in levels:
        if lvl['type'] != level_type:
            continue
        dist = abs(lvl['price'] - price)
        if dist <= tolerance and (closest is None or dist < closest_dist):
            closest = lvl
            closest_dist = dist
    if closest is None:
        levels.append({'price': float(price), 'type': level_type, 'weight': float(weight)})
    else:
        total_weight = closest['weight'] + weight
        closest['price'] = (closest['price'] * closest['weight'] + price * weight) / total_weight
        closest['weight'] = total_weight


def build_memory_from_day(prices, day_weight):
    lows, highs = detect_local_levels(prices)
    levels = []
    for price in lows:
        merge_level(levels, price, 'low', day_weight)
    for price in highs:
        merge_level(levels, price, 'high', day_weight)
    return levels


def get_sorted_levels(levels, level_type):
    filtered = [lvl for lvl in levels if lvl['type'] == level_type]
    return sorted(filtered, key=lambda l: l['price'])


def calculate_trade_quantity(price, reference_price, max_quantity=10, step=2.0):
    diff = price - reference_price
    if diff <= 0:
        return 1
    qty = int(diff // step) + 1
    if qty < 1:
        qty = 1
    return min(max_quantity, qty)


def simulate_day(day_df, memory_levels, max_position=12, tolerance=1.2):
    trades = []
    holdings = []
    buy_levels = get_sorted_levels(memory_levels, 'low')
    sell_levels = get_sorted_levels(memory_levels, 'high')
    used_buy_levels = set()
    last_buy_price = None
    last_price = None

    def find_candidate_buy(price):
        candidates = [
            (idx, lvl)
            for idx, lvl in enumerate(buy_levels)
            if idx not in used_buy_levels and price <= lvl['price'] + tolerance
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1]['price'])

    def find_candidate_sell(price, avg_buy):
        candidates = [lvl for lvl in sell_levels if price >= lvl['price'] - tolerance and lvl['price'] >= avg_buy + 0.1]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x['price'])

    for _, row in day_df.iterrows():
        price = row['mid_price']
        ts = row['abs_timestamp']
        last_price = price

        if holdings:
            avg_buy = sum(holdings) / len(holdings)
            sell_level = find_candidate_sell(price, avg_buy)
            if sell_level is not None:
                qty = len(holdings)
                pnl = (price - avg_buy) * qty
                trades.append({
                    'action': 'SELL',
                    'price': price,
                    'quantity': qty,
                    'timestamp': ts,
                    'avg_buy': avg_buy,
                    'pnl': pnl,
                    'sell_level': sell_level['price'],
                })
                holdings = []
                last_buy_price = None
                continue

        candidate = find_candidate_buy(price)
        if candidate is not None:
            idx, level = candidate
            if last_buy_price is None or price < last_buy_price + 0.05:
                quantity = calculate_trade_quantity(price, level['price'])
                quantity = min(quantity, max_position - len(holdings))
                if quantity > 0:
                    trades.append({
                        'action': 'BUY',
                        'price': price,
                        'quantity': quantity,
                        'timestamp': ts,
                        'buy_level': level['price'],
                    })
                    holdings.extend([price] * quantity)
                    used_buy_levels.add(idx)
                    last_buy_price = price

    if holdings and last_price is not None:
        avg_buy = sum(holdings) / len(holdings)
        pnl = (last_price - avg_buy) * len(holdings)
        trades.append({
            'action': 'CLOSE',
            'price': last_price,
            'quantity': len(holdings),
            'timestamp': day_df.iloc[-1]['abs_timestamp'],
            'avg_buy': avg_buy,
            'pnl': pnl,
            'note': 'end_of_day_close',
        })
    return trades


# Daten laden
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

# Memory initialisieren mit Day 0 Referenz
memory_levels = build_memory_from_day(price_dfs[0]['mid_price'].tolist(), day_weight=1.0)
print(f"Initiale Day 0 Levels: {len(memory_levels)}")

results = {}
def get_day_weight(day):
    # Day 0 = Basisgewicht 1, späteres Daten erhalten stärkeres Gewicht
    return 1.0 + day * 2.0

for day in [1, 2]:
    trades = simulate_day(price_dfs[day], memory_levels)
    total_pnl = sum(t['pnl'] for t in trades if t['action'] in ('SELL', 'CLOSE'))
    results[day] = {
        'trades': trades,
        'pnl': total_pnl,
        'total_buys': sum(1 for t in trades if t['action'] == 'BUY'),
        'total_sells': sum(1 for t in trades if t['action'] == 'SELL'),
        'end_closes': sum(1 for t in trades if t['action'] == 'CLOSE'),
    }

    # Update memory with this Day's levels: spätere Tage bekommen stärkeres Gewicht
    day_weight = get_day_weight(day)
    new_levels = build_memory_from_day(price_dfs[day]['mid_price'].tolist(), day_weight=day_weight)
    for lvl in new_levels:
        merge_level(memory_levels, lvl['price'], lvl['type'], lvl['weight'], tolerance=1.0)
    print(f"Day {day} hinzugefügte Levels: {len(new_levels)} (weight={day_weight}) => Memory jetzt: {len(memory_levels)}")

# Ausgabe
print("\n=== Strategie Ergebnis ===")
for day in [1, 2]:
    print(f"Day {day}: Buys={results[day]['total_buys']} Sells={results[day]['total_sells']} Close={results[day]['end_closes']} PnL={results[day]['pnl']:.2f}")

    if day == 2:
        print(f"Day 1 + Day 2 Total PnL: {results[1]['pnl'] + results[2]['pnl']:.2f}")

# Grafik erstellen
plt.figure(figsize=(14, 7))
for day, color in [(1, 'blue'), (2, 'orange')]:
    df = price_dfs[day]
    plt.plot(df['abs_timestamp'], df['mid_price'], marker='o', markersize=4, linestyle='-', label=f'Day {day}', color=color)
    for trade in results[day]['trades']:
        if trade['action'] == 'BUY':
            plt.scatter(trade['timestamp'], trade['price'], color='green', s=70, marker='^')
        elif trade['action'] == 'SELL':
            plt.scatter(trade['timestamp'], trade['price'], color='red', s=70, marker='v')
        elif trade['action'] == 'CLOSE':
            plt.scatter(trade['timestamp'], trade['price'], color='purple', s=70, marker='X')

buy_levels = get_sorted_levels(memory_levels, 'low')
sell_levels = get_sorted_levels(memory_levels, 'high')
for lvl in buy_levels:
    plt.axhline(lvl['price'], color='gray', linestyle='--', alpha=0.15)
for lvl in sell_levels:
    plt.axhline(lvl['price'], color='red', linestyle=':', alpha=0.15)

plt.title(f'Memory Weighted Strategie - Produkt {product}')
plt.xlabel('Absoluter Zeitstempel')
plt.ylabel('Mid Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path / 'memory_weighted_strategy.png', dpi=300)
print(f"Grafik gespeichert: {output_path / 'memory_weighted_strategy.png'}")
