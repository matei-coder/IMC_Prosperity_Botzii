import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Pfade zu den CSV-Dateien
round3_path = Path("/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/ROUND_3")

# Alle Price-Dateien laden und kombinieren
price_dfs = []
for day in range(3):
    df = pd.read_csv(round3_path / f"prices_round_3_day_{day}.csv", sep=";")
    price_dfs.append(df)

# Kombiniere alle Price-Daten
prices_df = pd.concat(price_dfs, ignore_index=True)

# Konvertiere notwendige Spalten zu numerischen Werten
prices_df['timestamp'] = pd.to_numeric(prices_df['timestamp'])
prices_df['mid_price'] = pd.to_numeric(prices_df['mid_price'])
prices_df['day'] = pd.to_numeric(prices_df['day'])

# Erstelle absoluten Timestamp (Tag * großer Wert + Timestamp)
prices_df['abs_timestamp'] = prices_df['day'] * 10000 + prices_df['timestamp']

print("Verfügbare Produkte:")
print(prices_df['product'].unique())
print("\n")

# ============================================
# Grafik 1: Preise aller Produkte über Zeit
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Round 3 - Preisanalyse', fontsize=16, fontweight='bold')

# Subplot 1: VELVETFRUIT_EXTRACT
ax = axes[0, 0]
vev_data = prices_df[prices_df['product'] == 'VELVETFRUIT_EXTRACT']
ax.plot(vev_data['abs_timestamp'], vev_data['mid_price'], marker='o', linewidth=2, markersize=4)
ax.set_title('VELVETFRUIT_EXTRACT', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Mid Price')
ax.grid(True, alpha=0.3)

# Subplot 2: HYDROGEL_PACK
ax = axes[0, 1]
hydro_data = prices_df[prices_df['product'] == 'HYDROGEL_PACK']
ax.plot(hydro_data['abs_timestamp'], hydro_data['mid_price'], marker='o', linewidth=2, markersize=4, color='orange')
ax.set_title('HYDROGEL_PACK', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Mid Price')
ax.grid(True, alpha=0.3)

# Subplot 3: VEV Voucher (kleine Werte)
ax = axes[1, 0]
vev_vouchers = prices_df[prices_df['product'].str.startswith('VEV_')]
for voucher in sorted(vev_vouchers['product'].unique()):
    vev_data = prices_df[prices_df['product'] == voucher]
    ax.plot(vev_data['abs_timestamp'], vev_data['mid_price'], marker='o', label=voucher, linewidth=1.5, markersize=3)
ax.set_title('VEV Voucher (Optionen)', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Mid Price')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Subplot 4: Alle Produkte zusammen (normalisiert)
ax = axes[1, 1]
for product in prices_df['product'].unique():
    product_data = prices_df[prices_df['product'] == product]
    prices = product_data['mid_price'].values
    normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1)
    ax.plot(product_data['abs_timestamp'], normalized, marker='o', label=product, linewidth=1.5, markersize=3)
ax.set_title('Alle Produkte (normalisiert)', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Normalisierter Preis')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/prices_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Grafik 1 gespeichert: prices_analysis.png\n")

# ============================================
# Grafik 2: PnL Strategie
# ============================================
# Strategie: Kaufe die günstigsten Voucher, verkaufe zu höheren Preisen

# Extrahiere Strike Prices aus den VEV Namen
vev_data = prices_df[prices_df['product'].str.startswith('VEV_')].copy()
vev_data['strike_price'] = vev_data['product'].str.extract(r'VEV_(\d+)').astype(int)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Round 3 - PnL Analyse (Voucher Strategie)', fontsize=16, fontweight='bold')

# Subplot 1: Voucher Preise nach Strike Price
ax = axes[0, 0]
for strike in sorted(vev_data['strike_price'].unique()):
    strike_data = vev_data[vev_data['strike_price'] == strike]
    ax.plot(strike_data['abs_timestamp'], strike_data['mid_price'], 
            marker='o', label=f'Strike {strike}', linewidth=2, markersize=4)
ax.set_title('Voucher Preise nach Strike Price', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Mid Price')
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

# Subplot 2: Spread zwischen niedrigstem und höchstem Strike
ax = axes[0, 1]
timestamps = sorted(prices_df['abs_timestamp'].unique())
spreads = []
for ts in timestamps:
    ts_data = vev_data[vev_data['abs_timestamp'] == ts]
    if len(ts_data) > 0:
        min_price = ts_data['mid_price'].min()
        max_price = ts_data['mid_price'].max()
        spread = max_price - min_price
        spreads.append(spread)
    else:
        spreads.append(0)

ax.plot(timestamps, spreads, marker='o', linewidth=2, markersize=5, color='green')
ax.set_title('Spread: Max Strike - Min Strike Preis', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Spread (Gewinn Potential)')
ax.grid(True, alpha=0.3)
ax.fill_between(timestamps, spreads, alpha=0.3, color='green')

# Subplot 3: PnL wenn man günstige Voucher kauft und teure verkauft
ax = axes[1, 0]
pnl_values = []
for ts in timestamps:
    ts_data = vev_data[vev_data['abs_timestamp'] == ts]
    if len(ts_data) > 0:
        # Kaufe 10 Stück des günstigsten Vouchers
        # Verkaufe sie als ob sie der teuerste Voucher wären
        min_price = ts_data['mid_price'].min()
        max_price = ts_data['mid_price'].max()
        
        # PnL pro Einheit = (max - min)
        # Gesamtposition = 10 Einheiten
        pnl = (max_price - min_price) * 10
        pnl_values.append(pnl)
    else:
        pnl_values.append(0)

cumulative_pnl = np.cumsum(pnl_values)
ax.plot(timestamps, pnl_values, marker='o', label='Einzelne PnL', linewidth=2, markersize=4, color='blue')
ax.plot(timestamps, cumulative_pnl, marker='s', label='Kumulativ', linewidth=2, markersize=4, color='red', linestyle='--')
ax.set_title('PnL: Kaufe min(Strike) Voucher, Verkaufe als max(Strike)', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('PnL')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Subplot 4: Detaillierte Statistiken
ax = axes[1, 1]
ax.axis('off')

stats_text = f"""
STRATEGIE STATISTIKEN

Zeitraum: 3 Tage (Round 3)
Anzahl Messpunkte: {len(timestamps)}

Voucher Preise:
  - Minimaler Preis: {vev_data['mid_price'].min():.2f}
  - Maximaler Preis: {vev_data['mid_price'].max():.2f}
  - Durchschnitt: {vev_data['mid_price'].mean():.2f}

Strike Prices: {', '.join(map(str, sorted(vev_data['strike_price'].unique())))}

Spread Statistiken:
  - Min Spread: {min(spreads):.2f}
  - Max Spread: {max(spreads):.2f}
  - Durchschn. Spread: {np.mean(spreads):.2f}

PnL (Position: 10 Einheiten):
  - Total PnL: {cumulative_pnl[-1]:.2f}
  - Durchschn. PnL pro Zeitpunkt: {np.mean(pnl_values):.2f}
  - Min PnL: {min(pnl_values):.2f}
  - Max PnL: {max(pnl_values):.2f}
"""

ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/pnl_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Grafik 2 gespeichert: pnl_analysis.png\n")

# ============================================
# Grafik 3: Detaillierte Vergleiche
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Round 3 - Detaillierte Analyse', fontsize=16, fontweight='bold')

# Subplot 1: VELVETFRUIT_EXTRACT vs Voucher Durchschnitt
ax = axes[0, 0]
vev_extract = prices_df[prices_df['product'] == 'VELVETFRUIT_EXTRACT']
voucher_avg = vev_data.groupby('abs_timestamp')['mid_price'].mean()

ax.plot(vev_extract['abs_timestamp'], vev_extract['mid_price'], 
        marker='o', label='VELVETFRUIT_EXTRACT', linewidth=2, markersize=5)
ax.plot(voucher_avg.index, voucher_avg.values, 
        marker='s', label='Voucher Ø', linewidth=2, markersize=5, linestyle='--')
ax.set_title('VELVETFRUIT_EXTRACT vs Voucher Durchschnitt', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Mid Price')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Preisverteilung nach Produkt
ax = axes[0, 1]
product_stats = []
product_names = []
for product in sorted(prices_df['product'].unique()):
    product_data = prices_df[prices_df['product'] == product]
    product_stats.append(product_data['mid_price'].values)
    product_names.append(product)

bp = ax.boxplot(product_stats, labels=product_names, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_title('Preisverteilung nach Produkt', fontweight='bold')
ax.set_ylabel('Mid Price')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Tägliche Durchschnitte
ax = axes[1, 0]
daily_avg = prices_df[prices_df['product'] == 'VELVETFRUIT_EXTRACT'].groupby('day')['mid_price'].agg(['mean', 'min', 'max'])
days = daily_avg.index
ax.bar(days, daily_avg['mean'], alpha=0.6, label='Durchschnitt', color='blue')
ax.fill_between(days, daily_avg['min'], daily_avg['max'], alpha=0.2, color='blue', label='Min-Max Range')
ax.set_title('VELVETFRUIT_EXTRACT - Tägliche Durchschnitte', fontweight='bold')
ax.set_xlabel('Tag')
ax.set_ylabel('Mid Price')
ax.set_xticks(days)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Subplot 4: Volatilität (Standardabweichung pro Zeitfenster)
ax = axes[1, 1]
volatility = []
vol_timestamps = []
for ts in sorted(prices_df['abs_timestamp'].unique()):
    ts_data = prices_df[prices_df['abs_timestamp'] == ts]
    std = ts_data['mid_price'].std()
    volatility.append(std)
    vol_timestamps.append(ts)

ax.plot(vol_timestamps, volatility, marker='o', linewidth=2, markersize=4, color='purple')
ax.fill_between(vol_timestamps, volatility, alpha=0.3, color='purple')
ax.set_title('Volatilität (Std Dev) über Zeit', fontweight='bold')
ax.set_xlabel('Zeit')
ax.set_ylabel('Standardabweichung')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/enriqueluca/IMC Prosperity Algorithm Round 3/IMC_Prosperity_Botzii/detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Grafik 3 gespeichert: detailed_analysis.png\n")

# ============================================
# Zusammenfassung ausdrucken
# ============================================
print("=" * 60)
print("ROUND 3 DATEN ANALYSE - ZUSAMMENFASSUNG")
print("=" * 60)

print("\n1. VERFÜGBARE PRODUKTE:")
for product in sorted(prices_df['product'].unique()):
    data = prices_df[prices_df['product'] == product]
    print(f"   {product:25} - Preis: {data['mid_price'].min():8.2f} - {data['mid_price'].max():8.2f} (Ø {data['mid_price'].mean():8.2f})")

print("\n2. VOUCHER ANALYSE (VEV):")
for strike in sorted(vev_data['strike_price'].unique()):
    strike_data = vev_data[vev_data['strike_price'] == strike]
    print(f"   VEV_{strike} - Preis: {strike_data['mid_price'].min():8.2f} - {strike_data['mid_price'].max():8.2f} (Ø {strike_data['mid_price'].mean():8.2f})")

print("\n3. STRATEGIE ERGEBNISSE:")
print(f"   Gesamter Zeitraum Spread: {min(spreads):.2f} - {max(spreads):.2f}")
print(f"   Durchschnittlicher Spread: {np.mean(spreads):.2f}")
print(f"   Kumulativer PnL (10er Position): {cumulative_pnl[-1]:.2f}")
print(f"   Durchschnittlicher PnL pro Zeitpunkt: {np.mean(pnl_values):.2f}")

print("\n4. GRAFIKEN ERSTELLT:")
print("   ✓ prices_analysis.png - Preisveränderungen")
print("   ✓ pnl_analysis.png - PnL Strategie")
print("   ✓ detailed_analysis.png - Detaillierte Vergleiche")

print("\n" + "=" * 60)
