from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas as pd
import matplotlib.pyplot as plt

# API-Key und Initialisierung des CryptoCurrencies-Objekts
api_key = 'L1XCRI6UYNUMGI0S'
cc = CryptoCurrencies(key=api_key, output_format='pandas')

# 1. Lade tägliche Bitcoin-Daten (BTC/USD)
data, meta = cc.get_digital_currency_daily(symbol='BTC', market='USD')

# Sortiere die Daten chronologisch und setze den Index als Spalte
data = data.sort_index().reset_index()

# Überprüfe die Spaltennamen
print("Spalten vor dem Umbenennen:", data.columns)

# Da die Datumsspalte als "date" vorliegt, benennen wir sie in "Date" um:
if 'date' in data.columns:
    data.rename(columns={'date': 'Date'}, inplace=True)
elif 'timestamp' in data.columns:
    data.rename(columns={'timestamp': 'Date'}, inplace=True)
else:
    data.rename(columns={'index': 'Date'}, inplace=True)

print("Spalten nach dem Umbenennen:", data.columns)

# Wandle die Datumsspalte in datetime-Objekte um
data['Date'] = pd.to_datetime(data['Date'])

# 2. Technische Indikatoren berechnen
# Da deine Daten die Spalte "4. close" enthalten, verwenden wir diese als Preis
price_col = '4. close'

# 200-Tage gleitender Durchschnitt (SMA)
data['SMA200'] = data[price_col].rolling(window=200).mean()

# EMA10, EMA50 und EMA200
data['EMA10']  = data[price_col].ewm(span=10, adjust=False).mean()
data['EMA50']  = data[price_col].ewm(span=50, adjust=False).mean()
data['EMA200'] = data[price_col].ewm(span=200, adjust=False).mean()

# Berechne den täglichen BTC-Rendite (prozentuale Änderung)
data['BTC_return'] = data[price_col].pct_change()

# 3. Portfolio-Simulation

# Bedingungen, bezogen auf den Vortag:
# Portfolio 2: BTC-Kurs am Vortag > 200-Tage-SMA am Vortag
data['cond_port2'] = (data[price_col].shift(1) > data['SMA200'].shift(1)).astype(int)

# Portfolio 3: EMA50 am Vortag > EMA200 am Vortag
data['cond_port3'] = (data['EMA50'].shift(1) > data['EMA200'].shift(1)).astype(int)

# Berechne die täglichen Renditen der Portfolios:
data['port1_return'] = data['BTC_return']                          # Portfolio 1: Einfach BTC-Rendite
data['port2_return'] = data['BTC_return'] * 2 * data['cond_port2']    # Portfolio 2: 2× BTC-Rendite, wenn Bedingung erfüllt, sonst 0
data['port3_return'] = data['BTC_return'] * 2 * data['cond_port3']    # Portfolio 3: 2× BTC-Rendite, wenn Bedingung erfüllt, sonst 0

# Ersetze NaN-Werte (z. B. in der ersten Zeile) mit 0
data[['port1_return', 'port2_return', 'port3_return']] = data[['port1_return', 'port2_return', 'port3_return']].fillna(0)

# Berechne den kumulativen Portfolio-Wert (Startwert = 100)
data['port1_value'] = 100 * (1 + data['port1_return']).cumprod()
data['port2_value'] = 100 * (1 + data['port2_return']).cumprod()
data['port3_value'] = 100 * (1 + data['port3_return']).cumprod()

# Ausgabe der ersten Zeilen zur Kontrolle
print(data[['Date', price_col, 'SMA200', 'EMA10', 'EMA50', 'EMA200', 'BTC_return',
            'port1_value', 'port2_value', 'port3_value']].head(10))

# Plot der Portfolio-Werte über die Zeit
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['port1_value'], label='Portfolio 1 (BTC Return)')
plt.plot(data['Date'], data['port2_value'], label='Portfolio 2 (2× Return, wenn BTC > SMA200)')
plt.plot(data['Date'], data['port3_value'], label='Portfolio 3 (2× Return, wenn EMA50 > EMA200)')
plt.xlabel('Datum')
plt.ylabel('Portfoliowert')
plt.title('Portfolio-Simulation basierend auf Bitcoin-Renditen')
plt.legend()
plt.tight_layout()
plt.show()

