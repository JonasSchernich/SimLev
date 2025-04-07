#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# ---------------------------------------------------
# Datenvorbereitung: S&P 500 Daten von Yahoo Finance einlesen
# ---------------------------------------------------
# Hole S&P 500 Daten von Yahoo Finance (Ticker: ^GSPC)
spx_data = yf.download("^GSPC", start="2000-01-01", end="2025-12-31")
spx_data = spx_data.reset_index()
spx = spx_data[["Date", "Close"]].rename(columns={"Close": "Price"})
spx = spx.sort_values("Date")

# Optional: Erstelle einen vollständigen täglichen Datumsbereich und interpoliere fehlende Werte
full_dates = pd.DataFrame({"Date": pd.date_range(start=spx["Date"].min(), end=spx["Date"].max(), freq="D")})
spx = full_dates.merge(spx, on="Date", how="left")
spx["Price"] = spx["Price"].interpolate(method="linear", limit_direction="both")
# Optional: DataFrame zuschneiden (z.B. ab Index 10000)
spx = spx.iloc[10000:].reset_index(drop=True)

# ---------------------------------------------------
# 1) Reverse Bonus Zertifikat Berechnung (Bear Case)
# ---------------------------------------------------
def price_reverse_bonus_certificate(S0, H, R, bonus, BR, r, sigma, T):
    K = R / BR
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = BR * (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    mu = (np.log(S0 / H)) / (sigma * np.sqrt(T))
    P_no_touch = 1 - np.exp(-2 * mu**2)
    V = np.exp(-r * T) * (P_no_touch * bonus) + (1 - P_no_touch) * put_price
    return V

# ---------------------------------------------------
# 2) Bonus Zertifikat Berechnung (Bull Case)
# ---------------------------------------------------
def price_bonus_certificate(S0, H, bonus, r, sigma, T):
    """
    Berechnet den Preis eines Bonus Zertifikats.
    Formel:
      V = S0 + bonus * exp(-r*T) * [N(d2) - (H/S0)^(2λ) * N(d2 - 2ln(S0/H)/(σ√T))]
    mit λ = (r + 0.5σ²)/σ² und
      d2 = (ln(S0/H) + (r - 0.5σ²)*T)/(σ√T).
    """
    lambda_val = (r + 0.5 * sigma**2) / (sigma**2)
    d2 = (np.log(S0 / H) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    digital_price = np.exp(-r * T) * (norm.cdf(d2) - (H / S0)**(2 * lambda_val) * norm.cdf(d2 - 2 * np.log(S0 / H) / (sigma * np.sqrt(T))))
    V = S0 + bonus * digital_price
    return V

# ---------------------------------------------------
# 3) Hilfsfunktion für Indikatoren
# ---------------------------------------------------
def calculate_indicator(series: pd.Series, window: int, method: str = "SMA") -> pd.Series:
    if method.upper() == "SMA":
        return series.rolling(window).mean()
    elif method.upper() == "EMA":
        return series.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("Method muss 'SMA' oder 'EMA' sein.")

# ---------------------------------------------------
# 4) Backtesting-Funktion mit dynamischer Restlaufzeit und Rebalancing
# ---------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    price_col: str,
    date_col: str,
    indicator_days: int,
    indicator_type: str,
    gap_factor: float,
    rbc_bonus: float,
    rbc_BR: float,
    rbc_r: float,
    rbc_sigma: float,
    rbc_maturity_in_years: float,  # wird initial übergeben, dann dynamisch überschrieben
    initial_capital: float,
    plot: bool,
    tax_rate: float,  # als Bruch, z.B. 0.25 für 25%
    # Für Bull Case:
    bull_metric: str,          # "preis" oder "tage"
    bull_thresholds: list,     # Schwellenwerte für den Wechsel der Bull Regime
    bull_allocations: list,    # Allokationen: [1x, 2x, 3x, fix, Bonus]
    # Für Bear Case:
    bear_thresholds: list,     # Schwellenwerte (Drawdown) für Bear Regime
    bear_allocations: list     # Allokationen: [1x, 2x, 3x, fix, Reverse Bonus]
):
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    indicator_raw = calculate_indicator(df[price_col], indicator_days, indicator_type)
    df["Indicator"] = gap_factor * indicator_raw
    df = df.dropna(subset=["Indicator"]).reset_index(drop=True)
    df["Regime"] = np.where(df[price_col] > df["Indicator"], "Bull", "Bear")
    df["RegimeSignal"] = df["Regime"]
    df["RegimeActive"] = df["RegimeSignal"].shift(1)
    df.loc[0, "RegimeActive"] = df.loc[0, "RegimeSignal"]
    df["RegimeChangeFlag"] = df["RegimeActive"] != df["RegimeActive"].shift(1)

    # Bear Case (Reverse Zertifikat)
    current_T_bear = 1.5  # Startwert in Jahren
    rollover_bear = False

    # Bull Case (Bonus Zertifikat)
    current_T_bull = 1.0

    # Portfolio-Initialisierung
    portfolio_value = initial_capital
    portfolio_values = [initial_capital]
    last_rebalance_value = initial_capital
    rebalance_count = 0

    # Sub-Regime-Indizes
    bull_regime_index = 1
    bear_regime_index = 1

    # Hilfsvariablen
    last_bear_low = None        # Tiefstkurs der letzten Bear-Phase
    last_bear_low_date = None   # Datum des Tiefstkurses der letzten Bear-Phase
    last_ath_price = None       # Letzter All-Time-High aus der Bull-Phase

    # Regime-Status (für den Plot)
    regime_state_list = [df.loc[0, "RegimeActive"]]

    for i in range(1, len(df)):
        row = df.loc[i]
        price_i = row[price_col]
        current_date = row[date_col]
        regime_active = row["RegimeActive"]

        if i == 1:
            if regime_active == "Bull":
                last_ath_price = df.loc[0, price_col]
            else:
                last_bear_low = df.loc[0, price_col]
                last_bear_low_date = df.loc[0, date_col]
                last_ath_price = df.loc[0, price_col]

        if row["RegimeChangeFlag"]:
            if regime_active == "Bear":  # Wechsel Bull -> Bear
                last_ath_price = df.loc[i-1, price_col]
                last_bear_low = price_i
                last_bear_low_date = current_date
                bull_regime_index = 1
                current_T_bull = 1.0
            else:  # Wechsel Bear -> Bull
                last_bear_low = None
                last_bear_low_date = None
                bull_regime_index = 1
                current_T_bull = 1.0
            bear_regime_index = 1
            current_T_bear = 1.5
            rollover_bear = False
            rebalance_count += 1

        if regime_active == "Bull":
            current_state = f"Bull R{bull_regime_index}"
        else:
            current_state = f"Bear R{bear_regime_index}"
        regime_state_list.append(current_state)

        price_yesterday = df.loc[i-1, price_col]
        spx_ret = (price_i / price_yesterday) - 1.0

        rebalancing = False

        if regime_active == "Bull":
            # Bull Case: Bonus Zertifikat
            T_cert = current_T_bull
            if len(bull_allocations) > 1:
                if bull_metric.lower() == "preis":
                    if last_bear_low is not None:
                        metric_value = (price_i - last_bear_low) / last_bear_low
                    else:
                        metric_value = 0.0
                elif bull_metric.lower() == "tage":
                    if last_bear_low_date is not None:
                        metric_value = (current_date - last_bear_low_date).days
                    else:
                        metric_value = 0.0
                else:
                    metric_value = 0.0
                if bull_regime_index < len(bull_allocations):
                    if metric_value >= bull_thresholds[bull_regime_index - 1]:
                        bull_regime_index += 1
                        rebalancing = True
                        rebalance_count += 1
                        current_T_bull = 1.0
                else:
                    bull_regime_index = 1
            else:
                bull_regime_index = 1

            allocation = bull_allocations[bull_regime_index - 1]
            bonus_component = 0.0
            if allocation[4] > 0:
                bonus_component = (price_bonus_certificate(price_i, row["Indicator"] * 1.1, rbc_bonus, rbc_r, rbc_sigma, T_cert) /
                                   price_bonus_certificate(price_yesterday, df.loc[i-1, "Indicator"] * 1.1, rbc_bonus, rbc_r, rbc_sigma, T_cert) - 1.0)
            daily_return = ((allocation[0] / 100) * spx_ret +
                            (allocation[1] / 100) * (2 * spx_ret) +
                            (allocation[2] / 100) * (3 * spx_ret) +
                            (allocation[3] / 100) * 0.0001 +
                            (allocation[4] / 100) * bonus_component)
            current_T_bull -= 1/365
            if current_T_bull < 0.5:
                current_T_bull = 1.0
                rebalancing = True
                rebalance_count += 1

        else:
            # Bear Case: Reverse Bonus Zertifikat
            T_cert = current_T_bear
            if last_ath_price is None:
                last_ath_price = df.loc[i-1, price_col]
            if (last_bear_low is None) or (price_i < last_bear_low):
                last_bear_low = price_i
                last_bear_low_date = current_date
            drawdown = (last_ath_price - price_i) / last_ath_price if last_ath_price is not None else 0.0
            if len(bear_allocations) > 1:
                if bear_regime_index < len(bear_allocations):
                    if drawdown >= bear_thresholds[bear_regime_index - 1]:
                        bear_regime_index += 1
                        rebalancing = True
                        rebalance_count += 1
                        current_T_bear = 1.5
                        rollover_bear = False
                else:
                    bear_regime_index = 1
            else:
                bear_regime_index = 1

            allocation = bear_allocations[bear_regime_index - 1]
            V_yesterday = price_reverse_bonus_certificate(
                S0=price_yesterday,
                H=df.loc[i-1, "Indicator"] * 1.1,
                R=65.0,
                bonus=rbc_bonus,
                BR=rbc_BR,
                r=rbc_r,
                sigma=rbc_sigma,
                T=T_cert
            )
            V_today = price_reverse_bonus_certificate(
                S0=price_i,
                H=row["Indicator"] * 1.1,
                R=65.0,
                bonus=rbc_bonus,
                BR=rbc_BR,
                r=rbc_r,
                sigma=rbc_sigma,
                T=T_cert
            )
            reverse_bonus_ret = (V_today / V_yesterday - 1.0) if V_yesterday != 0 else 0.0
            daily_return = ((allocation[0] / 100) * spx_ret +
                            (allocation[1] / 100) * (2 * spx_ret) +
                            (allocation[2] / 100) * (3 * spx_ret) +
                            (allocation[3] / 100) * 0.0001 +
                            (allocation[4] / 100) * reverse_bonus_ret)
            current_T_bear -= 1/365
            time_rollover_event_bear = False
            if current_T_bear < 1.4:
                current_T_bear = 1.5
                rebalancing = True
                time_rollover_event_bear = True
                rebalance_count += 1

        portfolio_value = portfolio_value * (1 + daily_return)

        if row["RegimeChangeFlag"] or rebalancing:
            gain = (portfolio_value / last_rebalance_value) - 1
            if regime_active == "Bear" and 'time_rollover_event_bear' in locals() and time_rollover_event_bear:
                portfolio_value = last_rebalance_value * (1 + gain)
            else:
                portfolio_value = last_rebalance_value * (1 + gain * (1 - tax_rate))
            last_rebalance_value = portfolio_value

        portfolio_values.append(portfolio_value)

    df["StrategyIndex"] = portfolio_values
    df["RegimeState"] = regime_state_list

    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df["StrategyIndex"], label="Strategie", zorder=5)
    benchmark = df[price_col] / df[price_col].iloc[0] * initial_capital
    plt.plot(df[date_col], benchmark, label="S&P 500 (ungehebelt)", zorder=5)
    plt.yscale("log")
    plt.xlabel("Datum")
    plt.ylabel("Portfoliowert")
    plt.title("Backtest Ergebnis")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    color_mapping = {
        "Bull R1": "lightgreen",
        "Bull R2": "green",
        "Bull R3": "darkgreen",
        "Bear R1": "mistyrose",
        "Bear R2": "red",
        "Bear R3": "darkred"
    }
    start_idx = 0
    for idx in range(1, len(df)):
        if df.loc[idx, "RegimeState"] != df.loc[idx-1, "RegimeState"]:
            start_date_seg = df.loc[start_idx, date_col]
            end_date_seg = df.loc[idx-1, date_col]
            state = df.loc[start_idx, "RegimeState"]
            plt.axvspan(start_date_seg, end_date_seg, color=color_mapping.get(state, "gray"), alpha=0.3)
            start_idx = idx
    plt.axvspan(df.loc[start_idx, date_col], df.loc[len(df)-1, date_col],
                color=color_mapping.get(df.loc[start_idx, "RegimeState"], "gray"), alpha=0.3)
    plt.show()

    print(f"Umschichtungen insgesamt: {rebalance_count}")

    return df

# ---------------------------------------------------
# Interaktive Parameterabfrage
# ---------------------------------------------------
def get_user_parameters():
    print("Bitte geben Sie die Parameter ein. Drücken Sie Enter, um den Standardwert zu übernehmen.")

    indicator_type = input("Welchen Indikator sollen wir nutzen? (EMA/SMA) [Standard: SMA]: ")
    if not indicator_type:
        indicator_type = "SMA"

    indicator_days = input("Auf wie viele Tage soll der Indikator basieren? [Standard: 200]: ")
    indicator_days = int(indicator_days) if indicator_days else 200

    gap_factor = input("Was soll der Gap sein? [Standard: 1.0]: ")
    gap_factor = float(gap_factor) if gap_factor else 1.0

    tax_rate = input("Geben Sie den Steuersatz für Umschichtungen in Prozent ein [Standard: 0]: ")
    tax_rate = float(tax_rate)/100 if tax_rate else 0.0

    bull_extra = input("Wie viele extra Bull Regime möchten Sie? (0 = nur Regime 1) [Standard: 0]: ")
    bull_extra = int(bull_extra) if bull_extra else 0
    bull_regime_count = 1 + bull_extra
    if bull_regime_count > 1:
        bull_metric = input("Welche Metrik soll für den Wechsel zu Bull Regimen verwendet werden? (Preis/Tage) [Standard: Preis]: ")
        if not bull_metric:
            bull_metric = "preis"
    else:
        bull_metric = "preis"

    bull_alloc_r1 = input("Bull Regime 1 – Prozentanteile (S&P 500 1x, 2x, 3x, fix return, Bonus Zertifikat) [Standard: 0,0,100,0,0]: ")
    if bull_alloc_r1:
        bull_alloc_r1 = [float(x.strip()) for x in bull_alloc_r1.split(",")]
    else:
        bull_alloc_r1 = [0, 0, 100, 0, 0]
    bull_allocations = [bull_alloc_r1]
    bull_thresholds = []
    for i in range(2, bull_regime_count+1):
        th = input(f"Geben Sie den Schwellenwert für den Wechsel von Bull Regime {i-1} zu {i} ein (abhängig von {bull_metric}) [Standard: 0.10]: ")
        th = float(th) if th else 0.10
        bull_thresholds.append(th)
        alloc = input(f"Bull Regime {i} – Prozentanteile (S&P 500 1x, 2x, 3x, fix return, Bonus Zertifikat) [Standard: 0,0,100,0,0]: ")
        if alloc:
            alloc = [float(x.strip()) for x in alloc.split(",")]
        else:
            alloc = [0, 0, 100, 0, 0]
        bull_allocations.append(alloc)

    bear_extra = input("Wie viele extra Bear Regime möchten Sie? (0 = nur Regime 1) [Standard: 0]: ")
    bear_extra = int(bear_extra) if bear_extra else 0
    bear_regime_count = 1 + bear_extra
    bear_alloc_r1 = input("Bear Regime 1 – Prozentanteile (S&P 500 1x, 2x, 3x, fix return, Reverse Bonus) [Standard: 0,0,0,0,100]: ")
    if bear_alloc_r1:
        bear_alloc_r1 = [float(x.strip()) for x in bear_alloc_r1.split(",")]
    else:
        bear_alloc_r1 = [0, 0, 0, 0, 100]
    bear_allocations = [bear_alloc_r1]
    bear_thresholds = []
    for i in range(2, bear_regime_count+1):
        th = input(f"Geben Sie den Drawdown-Schwellenwert für den Wechsel von Bear Regime {i-1} zu {i} ein [Standard: 0.25]: ")
        th = float(th) if th else 0.25
        bear_thresholds.append(th)
        alloc = input(f"Bear Regime {i} – Prozentanteile (S&P 500 1x, 2x, 3x, fix return, Reverse Bonus) [Standard: 100,0,0,0,0]: ")
        if alloc:
            alloc = [float(x.strip()) for x in alloc.split(",")]
        else:
            alloc = [100, 0, 0, 0, 0]
        bear_allocations.append(alloc)

    rbc_bonus = input("Geben Sie den Bonus für das Reverse Bonus Zertifikat ein [Standard: 10]: ")
    rbc_bonus = float(rbc_bonus) if rbc_bonus else 10.0
    rbc_BR = input("Geben Sie das Bezugsverhältnis ein [Standard: 0.01]: ")
    rbc_BR = float(rbc_BR) if rbc_BR else 0.01
    rbc_r = input("Geben Sie den risikofreien Zinssatz ein [Standard: 0.03]: ")
    rbc_r = float(rbc_r) if rbc_r else 0.03
    rbc_sigma = input("Geben Sie die Volatilität ein [Standard: 0.20]: ")
    rbc_sigma = float(rbc_sigma) if rbc_sigma else 0.20
    rbc_maturity = input("Geben Sie die Laufzeit in Jahren ein [Standard: 1.0]: ")
    rbc_maturity = float(rbc_maturity) if rbc_maturity else 1.0

    return {
        "indicator_type": indicator_type,
        "indicator_days": indicator_days,
        "gap_factor": gap_factor,
        "tax_rate": tax_rate,
        "bull_regime_count": bull_regime_count,
        "bull_metric": bull_metric,
        "bull_thresholds": bull_thresholds,
        "bull_allocations": bull_allocations,
        "bear_regime_count": bear_regime_count,
        "bear_thresholds": bear_thresholds,
        "bear_allocations": bear_allocations,
        "rbc_bonus": rbc_bonus,
        "rbc_BR": rbc_BR,
        "rbc_r": rbc_r,
        "rbc_sigma": rbc_sigma,
        "rbc_maturity": rbc_maturity
    }

# ---------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------
def main():
    params = get_user_parameters()

    result_df = run_backtest(
        df=spx,
        price_col="Price",
        date_col="Date",
        indicator_days=params["indicator_days"],
        indicator_type=params["indicator_type"],
        gap_factor=params["gap_factor"],
        rbc_bonus=params["rbc_bonus"],
        rbc_BR=params["rbc_BR"],
        rbc_r=params["rbc_r"],
        rbc_sigma=params["rbc_sigma"],
        rbc_maturity_in_years=params["rbc_maturity"],
        initial_capital=100.0,
        plot=True,
        tax_rate=params["tax_rate"],
        bull_metric=params["bull_metric"],
        bull_thresholds=params["bull_thresholds"],
        bull_allocations=params["bull_allocations"],
        bear_thresholds=params["bear_thresholds"],
        bear_allocations=params["bear_allocations"]
    )

    print(result_df.tail(10))

if __name__ == "__main__":
    main()
