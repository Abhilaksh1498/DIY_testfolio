#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data helper utilities for downloading and preparing external series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def download_fred_rate_series_to_index(series_id,
                                       output_path,
                                       start_date="1992-01-01",
                                       index_name="US 3M T-Bill Index",
                                       day_count=360):
    """
    Download a FRED rate series and convert it into an index-level series.

    Parameters:
    -----------
    series_id : str
        FRED series ID (e.g., 'DGS3MO' or 'TB3MS')
    output_path : str
        CSV output path
    start_date : str
        Start date in YYYY-MM-DD format
    index_name : str
        Name to use for the index (written as 'Index Value' column)
    day_count : int
        Day-count basis for converting annualized rate to daily rate
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["Date", "Rate"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.to_datetime(start_date)].copy()
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df.dropna(subset=["Rate"])

    daily_rate = df["Rate"] / 100.0 / day_count
    df[index_name] = 100.0 * (1.0 + daily_rate).cumprod()

    df_out = df[["Date", index_name]].rename(columns={index_name: "Index Value"})
    df_out.to_csv(output_path, index=False)
    return df_out


def download_fred_series(series_id, output_path=None, start_date=None):
    """
    Download a FRED series as a DataFrame.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=[series_id])
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)].copy()
    if output_path:
        df.to_csv(output_path, index=False)
    return df


def plot_cash_with_regimes(cash_index_csv,
                           start_date="1992-01-01",
                           end_date=None,
                           rate_series_id="DFF",
                           recession_series_id="USREC",
                           sp500_series_id="SP500",
                           save_path="/tmp/us_3m_tbill_regimes.png"):
    """
    Plot cash index with monetary regime shading and SP500 regime shading.
    """
    cash_df = pd.read_csv(cash_index_csv)
    cash_df["Date"] = pd.to_datetime(cash_df["Date"])
    if start_date:
        cash_df = cash_df[cash_df["Date"] >= pd.to_datetime(start_date)].copy()
    if end_date:
        cash_df = cash_df[cash_df["Date"] <= pd.to_datetime(end_date)].copy()

    rate_df = download_fred_series(rate_series_id, start_date=start_date)
    rec_df = download_fred_series(recession_series_id, start_date=start_date)
    sp_df = download_fred_series(sp500_series_id, start_date=start_date)

    # Align to cash dates
    cash_dates = pd.Index(cash_df["Date"])
    rate_series = rate_df.set_index("Date")[rate_series_id].reindex(cash_dates).ffill().bfill()
    rec_series = rec_df.set_index("Date")[recession_series_id].reindex(cash_dates).ffill().bfill()
    sp_series = sp_df.set_index("Date")[sp500_series_id].reindex(cash_dates).ffill().bfill()

    # Monetary regime: 6-month change in policy rate
    window = 126
    rate_change = rate_series - rate_series.shift(window)
    tightening = rate_change > 0.25
    easing = rate_change < -0.25

    # SP500 regime: 12-month price return
    sp_return = sp_series / sp_series.shift(252) - 1
    sp_rally = sp_return > 0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: cash index + monetary regime + recessions
    axes[0].plot(cash_df["Date"], cash_df["Index Value"], color="#1f77b4", linewidth=1.5)
    axes[0].set_title("US 3M T-Bill Index (Base 100) with Monetary Regimes", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Index Value")
    axes[0].grid(True, alpha=0.3)

    # Shade easing/tightening
    _shade_boolean_regions(axes[0], cash_dates, tightening, color="#f2a6a6", alpha=0.25, label="Tightening")
    _shade_boolean_regions(axes[0], cash_dates, easing, color="#a6d8f2", alpha=0.25, label="Easing")
    _shade_boolean_regions(axes[0], cash_dates, rec_series > 0, color="#b0b0b0", alpha=0.2, label="Recession")

    # Top legend
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)

    # Bottom: SP500 regimes
    axes[1].plot(cash_dates, sp_series.values, color="#2ca02c", linewidth=1.2)
    axes[1].set_title("S&P 500 Price Index with Rally/Underperform Regimes", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("S&P 500")
    axes[1].grid(True, alpha=0.3)

    _shade_boolean_regions(axes[1], cash_dates, sp_rally, color="#c9f2c9", alpha=0.25, label="Rally (12M > 0)")
    _shade_boolean_regions(axes[1], cash_dates, ~sp_rally, color="#f5c7c7", alpha=0.25, label="Underperform (12M <= 0)")
    axes[1].legend(loc="upper left", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    return save_path


def _shade_boolean_regions(ax, dates, mask, color, alpha=0.2, label=None):
    dates = pd.Index(dates)
    mask = pd.Series(mask, index=dates).fillna(False)
    if mask.any():
        # Find contiguous regions
        regions = []
        start = None
        for dt, flag in mask.items():
            if flag and start is None:
                start = dt
            if not flag and start is not None:
                regions.append((start, dt))
                start = None
        if start is not None:
            regions.append((start, dates[-1]))

        for i, (s, e) in enumerate(regions):
            ax.axvspan(s, e, color=color, alpha=alpha, label=label if i == 0 else None)
