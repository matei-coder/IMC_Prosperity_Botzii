import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

round3_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/ROUND_3")
output_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/Test Round 3 ")

# Laden aller Preise
price_dfs = []
for day in range(3):
    df = pd.read_csv(round3_path / f"prices_round_3_day_{day}.csv", sep=";")
    price_dfs.append(df)
prices_df = pd.concat(price_dfs, ignore_index=True)

prices_df['day'] = pd.to_numeric(prices_df['day'])
prices_df['timestamp'] = pd.to_numeric(prices_df['timestamp'])
prices_df['mid_price'] = pd.to_numeric(prices_df['mid_price'])
prices_df['abs_timestamp'] = prices_df['day'] * 10000 + prices_df['timestamp']

# Round-Zuordnung: 48h pro Round
prices_df['round'] = prices_df['day'] // 2

product = 'VEV_4000'
product_df = prices_df[prices_df['product'] == product].copy()
if product_df.empty:
    raise ValueError(f'Produkt {product} nicht gefunden.')

summary_rows = []
for round_number, group in product_df.groupby('round'):
    round_end = group.loc[group['abs_timestamp'].idxmax()]
    buy_point = group.loc[group['mid_price'].idxmin()]
    round_mean = group['mid_price'].mean()
    buy_vs_mean = (round_mean - buy_point['mid_price']) / round_mean * 100
    cumulative_before_buy = group[group['abs_timestamp'] <= buy_point['abs_timestamp']]['mid_price']
    rank = (cumulative_before_buy <= buy_point['mid_price']).mean() * 100
    pnl = (round_end['mid_price'] - buy_point['mid_price']) / buy_point['mid_price'] * 100
    summary_rows.append({
        'round': int(round_number),
        'buy_time': int(buy_point['day']),
        'buy_timestamp': int(buy_point['timestamp']),
        'buy_price': float(buy_point['mid_price']),
        'end_price': float(round_end['mid_price']),
        'pnl_pct': float(pnl),
        'buy_vs_round_mean_pct': float(buy_vs_mean),
        'buy_rank_pct': float(rank),
        'buy_time_abs': int(buy_point['abs_timestamp']),
        'end_time_abs': int(round_end['abs_timestamp']),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(product_df['abs_timestamp'], product_df['mid_price'], marker='o', label=f'{product} Mid Price')
for row in summary_rows:
    plt.scatter(row['buy_time_abs'], row['buy_price'], color='green', s=120, marker='^', label=f'Buy Round {row["round"]}' if row['round'] == 0 else None)
    plt.scatter(row['end_time_abs'], row['end_price'], color='red', s=120, marker='X', label=f'End Round {row["round"]}' if row['round'] == 0 else None)
    plt.plot([row['buy_time_abs'], row['end_time_abs']], [row['buy_price'], row['end_price']], color='gray', alpha=0.4)
    plt.annotate(
        f"Round {row['round']}:\nBuy {row['buy_price']:.2f}\nEnd {row['end_price']:.2f}\nPnL {row['pnl_pct']:.1f}%",
        xy=(row['end_time_abs'], row['end_price']), xytext=(5, 15), textcoords='offset points', fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5)
    )

# Round boundaries
round_boundaries = sorted(product_df.groupby('round')['abs_timestamp'].max().values)
for boundary in round_boundaries:
    plt.axvline(boundary, color='black', linestyle='--', alpha=0.4)
    plt.text(boundary + 50, product_df['mid_price'].min(), 'Round boundary', rotation=90, va='bottom', fontsize=9)

plt.title(f'Runde-halt Test: {product} - Kauf beim Tiefpunkt, Halten bis Rundenschluss')
plt.xlabel('Absoluter Zeitstempel')
plt.ylabel('Mid Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_path / 'round_end_hold_test.png', dpi=300)
print(f"Grafik gespeichert als {output_path / 'round_end_hold_test.png'}")

# Detaillierte Bewertung
for row in summary_rows:
    print(f"\nRound {row['round']}: Buy bei {row['buy_price']:.2f} am Tag {row['buy_time']} Timestamp {row['buy_timestamp']}")
    print(f"  - Preis war {row['buy_vs_round_mean_pct']:.1f}% unter dem Rundendurchschnitt")
    print(f"  - Kaufpreis war in den bisherigen Punkten im Round besser als {row['buy_rank_pct']:.1f}%")
    print(f"  - Ende der Runde: {row['end_price']:.2f}")
    print(f"  - Return bis Rundenschluss: {row['pnl_pct']:.1f}%")
