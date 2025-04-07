import pandas as pd
import matplotlib.pyplot as plt

# URL mit KGV-Daten des S&P 500 nach Monat
url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"

# Tabellen auf der Seite auslesen
tables = pd.read_html(url)

# Erste Tabelle enthält die KGV-Daten
df = tables[0]
df.columns = ["Date", "PE Ratio"]

# Datumsformat umwandeln und sortieren
df["Date"] = pd.to_datetime(df["Date"])
df["PE Ratio"] = pd.to_numeric(df["PE Ratio"], errors="coerce")
df = df.dropna()
df = df.sort_values("Date")

# Daten anzeigen
print(df.head())

# Optional: Plot der KGV-Daten
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["PE Ratio"])
plt.title("Historisches KGV des S&P 500")
plt.xlabel("Datum")
plt.ylabel("KGV")
plt.grid(True)
plt.tight_layout()
plt.show()
