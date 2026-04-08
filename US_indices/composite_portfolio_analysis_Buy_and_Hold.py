#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 COMPOSITE PORTFOLIO ANALYSIS TOOL
===============================================================================

Build and compare portfolios from multiple indices.
All functions are self‑contained – just place your CSV files in the working
directory (or a subfolder) and adjust the search path.

Author: Retail Investor
Date: February 2026
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
import warnings
from IPython.display import display, clear_output
from ipywidgets import interact, IntSlider, FloatSlider, VBox, Output, SelectionSlider
import matplotlib.dates as mdates
from US_indices.rebalance import build_rebalanced_portfolio_on_grid


warnings.filterwarnings('ignore')

# =============================================================================
# 1. HELPER FUNCTIONS (shared)
# =============================================================================

def parse_date(date_str):
    """Parse date strings in multiple formats."""
    formats = ['%d %b %Y', '%d-%b-%Y', '%Y-%m-%d']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return None


def calculate_xirr(flows, dates, guess=0.1):
    """
    Calculate XIRR using Newton‑Raphson method.

    Parameters:
    flows : list of cash flows (negative for outflows, positive for inflows)
    dates : list of datetime objects corresponding to flows
    guess : initial guess for the rate

    Returns:
    float: annualized internal rate of return
    """
    if len(flows) != len(dates):
        return None
    days = [(d - dates[0]).days for d in dates]

    def npv(rate):
        return sum(f / (1 + rate) ** (d / 365) for f, d in zip(flows, days))

    rate = guess
    for _ in range(100):
        f = npv(rate)
        f_prime = sum(-f * d / (365 * (1 + rate) ** (d / 365 + 1))
                      for f, d in zip(flows, days))
        if abs(f_prime) < 1e-10:
            break
        new_rate = rate - f / f_prime
        if abs(new_rate - rate) < 1e-8:
            return new_rate
        rate = new_rate
    return rate


def load_index_files(file_prefix, search_path='.', recursive=True, data_format='auto'):
    """
    Load CSV files matching the prefix.
    
    Parameters:
    data_format : str, one of:
        'auto' - auto-detect (default)
        'levels' - data is actual index levels
        'returns' - data is daily percentage returns (as fractions)
    """
    if recursive:
        pattern = os.path.join(search_path, '**', f'*{file_prefix}*.csv')
    else:
        pattern = os.path.join(search_path, f'*{file_prefix}*.csv')

    all_files = glob.glob(pattern, recursive=recursive)
    if not all_files:
        return None

    data_frames = []
    for fpath in sorted(all_files):
        try:
            df = pd.read_csv(fpath)
            df.columns = df.columns.str.strip()
            
            # Identify the value column
            value_col = None
            for col in df.columns:
                col_lower = col.lower()
                if ('total returns' in col_lower or 'index value' in col_lower or 
                    'nav' in col_lower or 'price' in col_lower or 'close' in col_lower):
                    value_col = col
                    break
            
            if value_col is None:
                value_col = df.columns[-1]   # fallback

            # Standardize column names
            df = df.rename(columns={df.columns[0]: 'Date', value_col: 'Raw_Value'})
            df['Date'] = df['Date'].apply(parse_date)
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date').drop_duplicates('Date')
            
            # Convert based on specified format
            if data_format == 'returns':
                # Convert daily returns to index levels (starting at 100)
                df['Index_Value'] = (1 + df['Raw_Value']).cumprod() * 100
                df['Index_Value'] = df['Index_Value'] / df['Index_Value'].iloc[0] * 100
            elif data_format == 'levels':
                df['Index_Value'] = df['Raw_Value']
            else:  # auto-detect
                sample = df['Raw_Value'].iloc[:10]
                # Check if values are small fractions (typically between -1 and 1)
                if sample.between(-1, 1).mean() > 0.8:
                    # Likely returns format
                    df['Index_Value'] = (1 + df['Raw_Value']).cumprod() * 100
                    df['Index_Value'] = df['Index_Value'] / df['Index_Value'].iloc[0] * 100
                else:
                    df['Index_Value'] = df['Raw_Value']
            
            df = df[['Date', 'Index_Value']].copy()
            # Added
            df = df.rename(columns={'Index_Value': value_col})
            data_frames.append(df)
            
        except Exception as e:
            print(f"   ⚠️  Error loading {os.path.basename(fpath)}: {e}")
            continue

    if not data_frames:
        return None

    combined = pd.concat(data_frames, ignore_index=True)
    combined = combined.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    return combined

# =============================================================================
# 2. PORTFOLIO CONSTRUCTION
# =============================================================================

def build_portfolio_series(portfolio_def, search_path='.', recursive=True, start_date=None, end_date=None, data_format='auto'):
    """
    Build a daily portfolio value series from a list of (index_name, weight) pairs.

    Parameters:
    portfolio_def : list of tuples, e.g. [("NIFTY500 MOMENTUM 50", 0.5), ...]
    search_path, recursive : passed to load_index_files
    start_date, end_date : optional date filters

    Returns:
    pd.DataFrame with columns 'Date' and 'Portfolio_Value' (aligned on common dates)
    """
    index_data = {}
    common_dates = None

    for idx_name, weight in portfolio_def:
        df = load_index_files(idx_name, search_path, recursive, data_format=data_format)
        if df is None or len(df) == 0:
            raise ValueError(f"Could not load data for {idx_name}")
        df = df.rename(columns={'Total Returns Index': idx_name})
        df = df.set_index('Date')
        index_data[idx_name] = df[idx_name]
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    # 2. Apply date filters (new)
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        # keep dates >= start_date
        common_dates = common_dates[common_dates >= start_date]
        if len(common_dates) == 0:
            raise ValueError(f"No data available on or after {start_date.date()}")
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        # keep dates <= end_date
        common_dates = common_dates[common_dates <= end_date]
        if len(common_dates) == 0:
            raise ValueError(f"No data available on or before {end_date.date()}")
    
    # Build combined DataFrame on common dates
    combined = pd.DataFrame(index=common_dates)
    for idx_name, weight in portfolio_def:
        combined[idx_name] = index_data[idx_name].loc[common_dates]

    # Normalize each index to 1 on the first common date
    first_date = common_dates[0]
    normalized = combined / combined.loc[first_date]
    weights = np.array([w for _, w in portfolio_def])
    portfolio_value = (normalized * weights).sum(axis=1)

    result = pd.DataFrame({
        'Date': common_dates,
        'Portfolio_Value': portfolio_value
    }).reset_index(drop=True)
    return result


# =============================================================================
# 3. PORTFOLIO ANALYSIS (metrics)
# =============================================================================

def analyze_portfolio(portfolio_name, portfolio_df,
                      risk_free_rate=6.0,
                      sip_amount=10000,
                      sip_day=2,
                      trading_days_per_year=252,
                      rolling_periods=[3, 5, 7, 10],
                      show_plots=True,
                      export_csv=False,
                      index_number=1):
    """
    Compute all metrics for a single portfolio.
    portfolio_df must have columns 'Date' and 'Portfolio_Value'.
    """
    df = portfolio_df.copy()
    df['Daily_Return'] = df['Portfolio_Value'].pct_change() * 100
    df = df.dropna().reset_index(drop=True)

    df['Cumulative_Return'] = (1 + df['Daily_Return'] / 100).cumprod()
    df['Running_Max'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] / df['Running_Max'] - 1) * 100

    # ---- Basic statistics ----
    start_val = df['Portfolio_Value'].iloc[0]
    end_val = df['Portfolio_Value'].iloc[-1]
    total_days = len(df)
    years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    total_return_pct = (end_val / start_val - 1) * 100
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    daily_returns = df['Daily_Return']

    # ---- Sharpe ratio ----
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days_per_year)
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0

    # ---- Rolling returns ----
    rolling_stats = {}
    for period in rolling_periods:
        rolling_days = int(period * trading_days_per_year)
        if len(df) > rolling_days:
            rolling_returns = []
            for i in range(rolling_days, len(df)):
                start = df.loc[i - rolling_days, 'Portfolio_Value']
                end = df.loc[i, 'Portfolio_Value']
                ann_return = ((end / start) ** (1 / period) - 1) * 100
                rolling_returns.append(ann_return)
            if rolling_returns:
                rolling_stats[f'{period}Y'] = {
                    'avg': np.mean(rolling_returns),
                    'median': np.median(rolling_returns),
                    'min': np.min(rolling_returns),
                    'max': np.max(rolling_returns),
                    'std': np.std(rolling_returns)
                }

    # ---- Max drawdown ----
    max_dd = df['Drawdown'].min()
    max_dd_date = df.loc[df['Drawdown'].idxmin(), 'Date'].date()

    # ---- SIP returns ----
    def calculate_sip_returns(df, amount, day):
        df_copy = df.copy()
        df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
        df_copy['Day'] = df_copy['Date'].dt.day

        sip_dates = []
        for month in df_copy['YearMonth'].unique():
            month_data = df_copy[df_copy['YearMonth'] == month]
            month_data['Day_Diff'] = abs(month_data['Day'] - day)
            closest = month_data.loc[month_data['Day_Diff'].idxmin()]
            sip_dates.append({
                'Date': closest['Date'],
                'Portfolio_Value': closest['Portfolio_Value']
            })

        sip_df = pd.DataFrame(sip_dates).sort_values('Date').reset_index(drop=True)
        total_investment = 0
        units = 0
        running_xirr = []
        running_dates = []

        for i, row in sip_df.iterrows():
            units += amount / row['Portfolio_Value']
            total_investment += amount
            current_value = units * row['Portfolio_Value']
            flows = [-amount] * (i + 1) + [current_value]
            dates = sip_df['Date'].iloc[:i + 1].tolist() + [row['Date']]
            xirr = calculate_xirr(flows, dates)
            if xirr is not None and np.isreal(xirr):
                if -0.99 <= xirr <= 10:          # keep only reasonable values
                    running_xirr.append(xirr * 100)
                    running_dates.append(row['Date'])

        final_value = units * sip_df['Portfolio_Value'].iloc[-1]
        return_pct = (final_value / total_investment - 1) * 100

        # Final XIRR (using all cash flows)
        cash_flows_final = [-amount] * len(sip_df) + [final_value]
        dates_final = sip_df['Date'].tolist() + [sip_df['Date'].iloc[-1]]
        xirr_final = calculate_xirr(cash_flows_final, dates_final)
        xirr_final_pct = xirr_final * 100 if xirr_final else None

        return {
            'return_pct': return_pct,
            'xirr': round(xirr_final_pct, 2) if xirr_final_pct else None,
            'installments': len(sip_df),
            'sip_start': sip_df['Date'].iloc[0].date(),
            'sip_end': sip_df['Date'].iloc[-1].date(),
            'xirr_series': pd.Series(running_xirr, index=running_dates),
                        # --- new lines ---
            'total_invested': total_investment,
            'final_value': final_value
        }

    sip_results = calculate_sip_returns(df, sip_amount, sip_day)

    # ---- Yearly returns (last 5 years) ----
    df['Year'] = df['Date'].dt.year
    yearly_returns = {}
    for year in sorted(df['Year'].unique(), reverse=True)[:5]:
        year_df = df[df['Year'] == year]
        if len(year_df) > 0:
            year_return = (year_df['Portfolio_Value'].iloc[-1] /
                           year_df['Portfolio_Value'].iloc[0] - 1) * 100
            yearly_returns[f'{year} Return %'] = year_return

    # ---- Summary dictionary ----
    summary = {
        'Portfolio': portfolio_name,
        'Date Range': f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        'Total Trading Days': len(df),
        'Start Value': round(start_val, 2),
        'End Value': round(end_val, 2),
        'Total Return %': round(total_return_pct, 2),
                # --- new lines ---
        'SIP Invested Amt': round(sip_results['total_invested'], 2),
        'SIP Final Amt': round(sip_results['final_value'], 2),
        'CAGR %': round(cagr, 2),
        'Annualized Volatility %': round(annual_vol, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown %': round(max_dd, 2),
        'Max Drawdown Date': max_dd_date,
        'SIP XIRR %': round(sip_results['xirr'], 2) if sip_results['xirr'] else None,
        'SIP Return %': round(sip_results['return_pct'], 2),
        'Positive Days %': round((daily_returns > 0).mean() * 100, 2),
        'Negative Days %': round((daily_returns < 0).mean() * 100, 2),
        'Max Daily Return %': round(daily_returns.max(), 2),
        'Min Daily Return %': round(daily_returns.min(), 2)
    }

    # Add rolling stats
    for period in rolling_periods:
        key = f'{period}Y'
        if key in rolling_stats:
            stats = rolling_stats[key]
            summary[f'{period}Y Rolling Avg %'] = round(stats['avg'], 2)
            summary[f'{period}Y Rolling Median %'] = round(stats['median'], 2)
            summary[f'{period}Y Rolling Min %'] = round(stats['min'], 2)
            summary[f'{period}Y Rolling Max %'] = round(stats['max'], 2)

    # Add yearly returns
    summary.update(yearly_returns)

    # Optional: plot single portfolio
    if show_plots:
        plot_single_portfolio(df, portfolio_name, index_number, rolling_periods, rolling_stats)

    # Optional: export data
    if export_csv:
        df.to_csv(f'{portfolio_name.replace(" ", "_")}_data.csv', index=False)
        print(f"   📁 Exported: {portfolio_name.replace(' ', '_')}_data.csv")

    return {
        'summary': summary,
        'data': df,
        'rolling_stats': rolling_stats,
        'sip_xirr_series': sip_results.get('xirr_series')
    }


# =============================================================================
# 4. PLOTTING FUNCTIONS (for portfolios)
# =============================================================================

def plot_single_portfolio(df, portfolio_name, index_number, rolling_periods, rolling_stats):
    """Four‑panel plot for a single portfolio."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{index_number}. {portfolio_name}', fontsize=16, fontweight='bold')

    # Portfolio Value
    axes[0, 0].plot(df['Date'], df['Portfolio_Value'], linewidth=1.5, color='#1f77b4')
    axes[0, 0].set_title('Portfolio Value', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)

    # Daily Returns Distribution
    axes[0, 1].hist(df['Daily_Return'], bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
    axes[0, 1].axvline(df['Daily_Return'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {df['Daily_Return'].mean():.3f}%")
    axes[0, 1].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative Returns
    axes[1, 0].plot(df['Date'], df['Cumulative_Return'], linewidth=1.5, color='#ff7f0e')
    axes[1, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Cumulative Return (x times)')
    axes[1, 0].grid(True, alpha=0.3)

    # Drawdown
    axes[1, 1].fill_between(df['Date'], 0, df['Drawdown'], color='red', alpha=0.5)
    axes[1, 1].plot(df['Date'], df['Drawdown'], color='darkred', linewidth=0.8)
    axes[1, 1].axhline(y=df['Drawdown'].min(), color='black', linestyle='--',
                       linewidth=1, label=f'Max DD: {df["Drawdown"].min():.1f}%')
    axes[1, 1].set_title('Drawdown', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_portfolio_comparison(all_data, all_results, rolling_periods):
    """Comparison dashboard for multiple portfolios."""
    num = len(all_data)
    if num < 1:
        return
    colors = plt.cm.Set1(np.linspace(0, 1, num))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Portfolio Comparison Dashboard', fontsize=16, fontweight='bold')

    # Normalized performance
    ax1 = axes[0, 0]
    for idx, (name, ddict) in enumerate(all_data.items()):
        df = ddict['data']
        if df is not None:
            norm = df['Portfolio_Value'] / df['Portfolio_Value'].iloc[0] * 100
            ax1.plot(df['Date'], norm, linewidth=1.5, color=colors[idx], label=name[:30])
            rebalance_dates = ddict.get('rebalance_dates')
            if rebalance_dates:
                rebalance_series = df.set_index('Date')['Portfolio_Value']
                mark_dates = [d for d in rebalance_dates if d in rebalance_series.index]
                if mark_dates:
                    mark_vals = (rebalance_series.loc[mark_dates] /
                                 df['Portfolio_Value'].iloc[0] * 100)
                    ax1.scatter(mark_dates, mark_vals, color=colors[idx],
                                s=18, alpha=0.7, marker='x')
    ax1.set_title('Normalized Performance (Start = 100)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Value')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # CAGR
    ax2 = axes[0, 1]
    names = list(all_results.keys())
    cagrs = [all_results[n]['CAGR %'] for n in names]
    bar_colors = ['green' if x > 0 else 'red' for x in cagrs]
    bars = ax2.bar(range(len(names)), cagrs, color=bar_colors, alpha=0.7)
    ax2.set_title('CAGR Comparison')
    ax2.set_xlabel('Portfolio')
    ax2.set_ylabel('CAGR (%)')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    for i, (bar, val) in enumerate(zip(bars, cagrs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Sharpe Ratio
    ax3 = axes[1, 0]
    sharpes = [all_results[n]['Sharpe Ratio'] for n in names]
    bars = ax3.bar(range(len(names)), sharpes, color='purple', alpha=0.7)
    ax3.set_title('Sharpe Ratio Comparison')
    ax3.set_xlabel('Portfolio')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Max Drawdown
    ax4 = axes[1, 1]
    drawdowns = [abs(all_results[n]['Max Drawdown %']) for n in names]
    bars = ax4.bar(range(len(names)), drawdowns, color='red', alpha=0.7)
    ax4.set_title('Max Drawdown Comparison')
    ax4.set_xlabel('Portfolio')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    for i, (bar, val) in enumerate(zip(bars, drawdowns)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Rolling returns heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    rolling_data = []
    for name in names:
        row = [name[:20]]
        for p in rolling_periods:
            col = f'{p}Y Rolling Median %'
            row.append(all_results[name].get(col, np.nan))
        rolling_data.append(row)
    columns = ['Portfolio'] + [f'{p}Y Median' for p in rolling_periods]
    rolling_df = pd.DataFrame(rolling_data, columns=columns).set_index('Portfolio')
    sns.heatmap(rolling_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                linewidths=0.5, cbar_kws={'label': 'Rolling Return (%)'})
    ax.set_title('Rolling Returns Comparison (Median %)')
    plt.tight_layout()
    plt.show()


def plot_running_xirr_comparison(all_data, all_results):
    """Plot the evolution of SIP XIRR for each portfolio."""
    if not all_data:
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))

    for idx, (name, data_dict) in enumerate(all_data.items()):
        xirr_series = data_dict.get('sip_xirr')
        if xirr_series is not None and len(xirr_series) > 0:
            # Optional: filter early noisy period
            # xirr_series = xirr_series[xirr_series.index > pd.Timestamp('2016-01-01')]
            ax.plot(xirr_series.index, xirr_series.values,
                    linewidth=2, color=colors[idx], label=name[:30])

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Running SIP XIRR Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('XIRR (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def build_portfolio_series_on_grid(portfolio_def, index_data, common_dates):
    """
    Build a portfolio using pre‑loaded index data and a fixed date grid.
    """
    combined = pd.DataFrame(index=common_dates)
    for idx_name, weight in portfolio_def:
        combined[idx_name] = index_data[idx_name].loc[common_dates]

    # Normalize each index to 1 on the first date
    first_date = common_dates[0]
    normalized = combined / combined.loc[first_date]
    weights = np.array([w for _, w in portfolio_def])
    portfolio_value = (normalized * weights).sum(axis=1)

    result = pd.DataFrame({
        'Date': common_dates,
        'Portfolio_Value': portfolio_value
    }).reset_index(drop=True)
    return result

def plot_running_cagr_comparison(all_data, all_results, period_years, trading_days_per_year=252,
                                 show_percentiles=False, rolling_percentiles_to_show=[5, 25, 50, 75, 95]):
    """
    Plot running X‑year CAGR for one or more portfolios.
    
    Percentile behavior:
    - Single portfolio: Shows percentiles for that portfolio
    - Multiple portfolios: Shows percentiles for combined data (all portfolios together)
                          Warning is printed when percentiles are shown with multiple portfolios
    
    Percentile labels are placed directly on the horizontal lines (no legend entries)
    All percentile lines use dashed black style
    
    Parameters:
    -----------
    show_percentiles : bool, default=False
        Whether to display horizontal percentile lines
    rolling_percentiles_to_show : list, default=[5, 25, 50, 75, 95]
        Percentiles to plot as horizontal lines
    """
    if not all_data:
        print("⚠️  No data to plot.")
        return

    # Collect all portfolio dataframes
    data_list = []
    for name, ddict in all_data.items():
        df = ddict.get('data')
        if df is not None and len(df) > 0:
            data_list.append((name, df))

    if len(data_list) == 0:
        print("⚠️  No valid portfolio data for running CAGR plot.")
        return

    num_portfolios = len(data_list)
    is_single_portfolio = (num_portfolios == 1)
    
    # Print warning if showing percentiles with multiple portfolios
    if show_percentiles and not is_single_portfolio:
        print(f"⚠️  WARNING: Showing percentiles for {num_portfolios} portfolios")
        print(f"   Percentile lines will be calculated on COMBINED data from all portfolios")
        print(f"   Individual portfolio percentiles are not displayed to avoid clutter\n")

    # Determine common date range across all portfolio data
    start_dates = [df['Date'].min() for _, df in data_list]
    end_dates   = [df['Date'].max() for _, df in data_list]
    common_start = max(start_dates)
    common_end   = min(end_dates)

    if common_start > common_end:
        print("⚠️  No overlapping date range for running CAGR plot.")
        return

    print(f"   📅 Common CAGR period ({period_years}Y): {common_start.date()} to {common_end.date()}")

    # Compute rolling CAGR for each portfolio on the common grid
    window_days = int(period_years * trading_days_per_year)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, num_portfolios))
    
    # Store all rolling returns for percentile calculation
    all_rolling_returns = []
    plotted_portfolios = []
    
    # First pass: collect all rolling returns and plot lines
    for idx, (name, df_full) in enumerate(data_list):
        # Slice to common dates
        mask = (df_full['Date'] >= common_start) & (df_full['Date'] <= common_end)
        df = df_full.loc[mask].copy().reset_index(drop=True)

        if len(df) <= window_days:
            print(f"   ⚠️  {name}: insufficient data for {period_years}‑year rolling CAGR (need {window_days} days, have {len(df)} days).")
            continue

        rolling_cagr = []
        rolling_dates = []

        for i in range(window_days, len(df)):
            start_val = df.loc[i - window_days, 'Portfolio_Value']
            end_val   = df.loc[i, 'Portfolio_Value']
            cagr = ((end_val / start_val) ** (1 / period_years) - 1) * 100
            rolling_cagr.append(cagr)
            rolling_dates.append(df.loc[i, 'Date'])

        if rolling_cagr:
            plotted_portfolios.append((name, rolling_dates, rolling_cagr, colors[idx]))
            ax.plot(rolling_dates, rolling_cagr, linewidth=2, color=colors[idx], label=name[:30], alpha=0.8)
            
            # Collect for combined percentiles (ONCE per portfolio, outside the inner loop)
            all_rolling_returns.extend(rolling_cagr)

    # Add horizontal percentile lines with direct labels (no legend entries)
    if show_percentiles and all_rolling_returns:
        percentile_values = {}
        print(f"\n   📊 {period_years}Y Rolling Returns Percentiles:")
        
        # Get the x-axis range for placing labels
        if plotted_portfolios:
            # Use the first portfolio's date range for label placement
            first_dates = plotted_portfolios[0][1]
            if first_dates:
                label_x = first_dates[0]  # Place at left edge
            else:
                label_x = common_start
        else:
            label_x = common_start
        
        # Calculate and plot percentiles
        for p in sorted(rolling_percentiles_to_show):
            value = np.percentile(all_rolling_returns, p)
            percentile_values[p] = value
            
            # Draw dashed black horizontal line
            ax.axhline(y=value, color='black', linestyle='--', 
                       linewidth=1.5, alpha=0.6)
            
            # Add label directly on the line
            # Alternate label position above/below to avoid overlap
            if p <= 50:
                va = 'bottom'
                y_offset = 0.3
            else:
                va = 'top'
                y_offset = -0.3
            
            ax.text(label_x, value + y_offset, f'{p}th ({value:.1f}%)', 
                    fontsize=9, color='black', va=va, ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
            
            print(f"      {p}th: {value:.2f}%")
        
        print(f"      Source: Combined data from {num_portfolios} portfolio(s)")
        print(f"      Total observations: {len(all_rolling_returns):,}")

    if not plotted_portfolios:
        print(f"⚠️  No portfolios had sufficient data for {period_years}‑year rolling CAGR.")
        ax.text(0.5, 0.5, f'Insufficient data for {period_years}‑year CAGR',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc='best', fontsize=10)
    
    # Add title with context about percentiles
    if show_percentiles and not is_single_portfolio:
        title = f'Running {period_years}‑Year CAGR (Combined Percentiles from {num_portfolios} Portfolios)'
    else:
        title = f'Running {period_years}‑Year CAGR (Common Period)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{period_years}‑Y CAGR (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_running_cagr_comparison_interactive(all_data, all_results, trading_days_per_year=252,
                                             show_percentiles=False, rolling_percentiles_to_show=[5, 25, 50, 75, 95],
                                             available_periods=[2.5, 3, 4, 5, 7, 10]):
    """
    Interactive plot of running X‑year CAGR with selection slider.
    Slider snaps to only the precomputed periods (can be irregularly spaced).
    """
    if not all_data:
        print("⚠️  No data to plot.")
        return

    # Collect all portfolio dataframes
    data_list = []
    for name, ddict in all_data.items():
        df = ddict.get('data')
        if df is not None and len(df) > 0:
            data_list.append((name, df))

    if len(data_list) == 0:
        print("⚠️  No valid portfolio data for running CAGR plot.")
        return

    num_portfolios = len(data_list)
    is_single_portfolio = (num_portfolios == 1)
    
    # Sort periods for consistent order
    sorted_periods = sorted(available_periods)
    min_period = min(sorted_periods)
    max_period = max(sorted_periods)
    
    # Determine common date range across all portfolio data
    start_dates = [df['Date'].min() for _, df in data_list]
    end_dates   = [df['Date'].max() for _, df in data_list]
    common_start = max(start_dates)
    common_end   = min(end_dates)
    
    # Pre-compute rolling returns for all periods
    print("📊 Precomputing rolling returns for all periods...")
    rolling_data_cache = {}
    
    for period_years in sorted_periods:
        print(f"   Computing {period_years}Y rolling returns...")
        
        window_days = int(period_years * trading_days_per_year)
        
        # Store rolling returns for each portfolio
        portfolio_rolling_data = []
        all_rolling_returns = []
        
        for name, df_full in data_list:
            mask = (df_full['Date'] >= common_start) & (df_full['Date'] <= common_end)
            df = df_full.loc[mask].copy().reset_index(drop=True)
            
            if len(df) <= window_days:
                print(f"   ⚠️  {name}: insufficient data for {period_years}‑year rolling CAGR")
                continue
            
            rolling_cagr = []
            rolling_dates = []
            
            for i in range(window_days, len(df)):
                start_val = df.loc[i - window_days, 'Portfolio_Value']
                end_val   = df.loc[i, 'Portfolio_Value']
                cagr = ((end_val / start_val) ** (1 / period_years) - 1) * 100
                rolling_cagr.append(cagr)
                rolling_dates.append(df.loc[i, 'Date'])
            
            if rolling_cagr:
                portfolio_rolling_data.append((name, rolling_dates, rolling_cagr))
                all_rolling_returns.extend(rolling_cagr)
        
        rolling_data_cache[period_years] = {
            'common_start': common_start,
            'common_end': common_end,
            'portfolio_data': portfolio_rolling_data,
            'all_rolling_returns': all_rolling_returns,
            'window_days': window_days,
            'num_portfolios': len(portfolio_rolling_data)
        }
    
    print("✅ Precomputation complete!\n")
    
    # Define the plotting function
    def plot_period(period_years):
        """Plot running CAGR for the selected period"""
        
        # Get cached data for this period
        cache = rolling_data_cache.get(period_years)
        if not cache:
            print(f"⚠️  No data available for {period_years}Y period")
            return
        
        portfolio_data = cache['portfolio_data']
        all_rolling_returns = cache['all_rolling_returns']
        
        if not portfolio_data:
            print(f"⚠️  No portfolio data available for {period_years}Y period")
            return
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(portfolio_data)))
        
        # Plot each portfolio
        for idx, (name, rolling_dates, rolling_cagr) in enumerate(portfolio_data):
            ax.plot(rolling_dates, rolling_cagr, linewidth=2, color=colors[idx], 
                    label=name[:30], alpha=0.8)
        
        # Add percentile lines if requested
        if show_percentiles and all_rolling_returns:
            print(f"\n   📊 {period_years}Y Rolling Returns Percentiles:")
            
            # Get label position (left edge)
            label_x = portfolio_data[0][1][0]
            
            # Plot percentiles
            for p in sorted(rolling_percentiles_to_show):
                value = np.percentile(all_rolling_returns, p)
                
                # Draw dashed black horizontal line
                ax.axhline(y=value, color='black', linestyle='--', 
                           linewidth=1.5, alpha=0.6)
                
                # Add label
                if p <= 50:
                    va = 'bottom'
                    y_offset = 0.3
                else:
                    va = 'top'
                    y_offset = -0.3
                
                ax.text(label_x, value + y_offset, f'{p}th ({value:.1f}%)', 
                        fontsize=9, color='black', va=va, ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 alpha=0.8, edgecolor='none'))
                
                print(f"      {p}th: {value:.2f}%")
            
            print(f"      Total observations: {len(all_rolling_returns):,}")
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add legend
        if portfolio_data:
            ax.legend(loc='best', fontsize=10)
        
        # Add title
        if show_percentiles and not is_single_portfolio:
            title = f'Running {period_years}‑Year CAGR (Combined Percentiles from {len(portfolio_data)} Portfolios)'
        else:
            title = f'Running {period_years}‑Year CAGR (Common Period)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{period_years}‑Y CAGR (%)')
        
        # ===== GRIDLINES =====
        # Set major ticks (every 2 years)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Set minor ticks (every 6 months)
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
        
        # Enable gridlines
        ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.grid(True, which='minor', axis='x', alpha=0.2, linestyle=':', linewidth=0.5)
        
        # Optional: Minor gridlines on y-axis
        from matplotlib.ticker import AutoMinorLocator
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', axis='y', alpha=0.15, linestyle=':', linewidth=0.5)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # ====================
        
        plt.tight_layout()
        plt.show()
    
    # Create the SelectionSlider that snaps to irregular values
    print("=" * 80)
    print(" INTERACTIVE RUNNING CAGR PLOT")
    print("=" * 80)
    print(f"Rolling periods available: {sorted_periods}")
    print("Use the slider below to select the rolling period")
    print("-" * 80)
    
    # Create SelectionSlider with irregularly spaced values
    selection_slider = SelectionSlider(
        options=sorted_periods,
        value=min_period,
        description='Rolling Period (years):',
        style={'description_width': 'initial'},
        layout={'width': '500px'}
    )
    
    # Display the interactive widget
    interact(plot_period, period_years=selection_slider)

def display_rolling_returns_summary(all_data, rolling_periods=[3, 5, 7, 10], 
                                    percentiles=[5, 25, 50, 75, 95], 
                                    trading_days_per_year=252):
    """
    Display a consolidated summary table of rolling X-year returns percentiles.
    """
    print("\n" + "=" * 100)
    print(" ROLLING X-YEAR RETURNS PERCENTILE SUMMARY")
    print("=" * 100)
    
    # Collect all results
    all_rows = []
    
    for period in rolling_periods:
        results = calculate_rolling_returns_percentiles(all_data, period, percentiles, trading_days_per_year)
        
        if not results:
            continue
        
        for name, stats in results.items():
            row = {
                'Portfolio': name[:40],
                'Period (Yrs)': period,
                'Count': stats['count'],
                'Mean %': round(stats['mean'], 2),
                'Median %': round(stats['median'], 2),
                'Std %': round(stats['std'], 2)
            }
            for p in percentiles:
                row[f'{p}th %'] = round(stats['percentiles'][f'p{p}'], 2)
            all_rows.append(row)
    
    if not all_rows:
        print("   No data available for rolling returns calculation")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Reorder columns for better readability
    column_order = ['Portfolio', 'Period (Yrs)', 'Count', 'Mean %', 'Median %', 'Std %'] + [f'{p}th %' for p in percentiles]
    df = df[column_order]
    
    # Display the DataFrame
    print("\n📊 CONSOLIDATED ROLLING RETURNS SUMMARY")
    print(df.to_string(index=False))
    
    # Add brief interpretation
    print("\n" + "=" * 100)
    # print("Interpretation: These percentiles show the distribution of X-year rolling returns.")
    # print(f"  • {percentiles[0]}th percentile: {percentiles[0]}% of periods had returns BELOW this value")
    # print(f"  • 50th (Median): Typical return for the period")
    # print(f"  • {percentiles[-1]}th percentile: {percentiles[-1]}% of periods had returns ABOVE this value")
    print("=" * 100)

def calculate_rolling_returns_percentiles(all_data, period_years, percentiles=[5, 25, 50, 75, 95], trading_days_per_year=252):
    """
    Calculate rolling returns percentiles across all portfolios.
    
    Parameters:
    -----------
    all_data : dict
        Dictionary containing portfolio data
    period_years : int
        Rolling period in years
    percentiles : list
        List of percentiles to compute (e.g., [5, 25, 50, 75, 95])
    trading_days_per_year : int, default=252
        Number of trading days in a year
    
    Returns:
    --------
    dict: Dictionary with percentiles for each portfolio and overall
    """
    window_days = int(period_years * trading_days_per_year)
    results = {}
    
    for name, ddict in all_data.items():
        df = ddict.get('data')
        if df is None or len(df) <= window_days:
            print(f"   ⚠️  {name}: insufficient data for {period_years}‑year rolling returns")
            continue
        
        rolling_returns = []
        for i in range(window_days, len(df)):
            start_val = df.loc[i - window_days, 'Portfolio_Value']
            end_val = df.loc[i, 'Portfolio_Value']
            ann_return = ((end_val / start_val) ** (1 / period_years) - 1) * 100
            rolling_returns.append(ann_return)
        
        if rolling_returns:
            results[name] = {
                'data': rolling_returns,
                'percentiles': {f'p{p}': np.percentile(rolling_returns, p) for p in percentiles},
                'count': len(rolling_returns),
                'mean': np.mean(rolling_returns),
                'median': np.median(rolling_returns),
                'std': np.std(rolling_returns)
            }
    
    return results

# =============================================================================
# 5. MAIN FUNCTION: compare multiple portfolios
# =============================================================================


def compare_portfolios(portfolio_list,
                       risk_free_rate=6.0,
                       sip_amount=10000,
                       sip_day=2,
                       trading_days_per_year=252,
                       rolling_periods=[3, 5, 7, 10],
                       show_plots=True,
                       export_csv=False,
                       plot_comparison=True,
                       search_path='.',
                       recursive=True,
                       output_fname="",
                       start_date=None,
                       end_date=None,
                       data_format = 'auto',
                       plot_running_xirr=False,
                       plot_running_cagr = True,
                       running_cagr_periods=[3],
                       plot_rolling_returns_percentiles=True,
                       rolling_percentiles_to_show=[5, 10, 50, 90, 95],
                       display_rolling_returns_table=True,
                       interactive_cagr=False,
                       rebalance=None,
                       tax_config=None,
                       plot_rebalance_marks=False,
                       show_rebalance_history=False
                       ):
    """
    Compare multiple composite portfolios using a common date grid.
    All portfolios are evaluated on exactly the same set of dates.
    """
    def resolve_tax_config(cfg, port_name, idx, total_ports):
        if cfg is None:
            return None
        if isinstance(cfg, (list, tuple)):
            if len(cfg) != total_ports:
                raise ValueError("tax_config list must match number of portfolios")
            return cfg[idx]
        if isinstance(cfg, dict):
            tax_keys = {'stcg_rate', 'ltcg_rate', 'stcg_days', 'ltcg_days', 'ltcg_exemption', 'apply_ltcg_exemption'}
            if tax_keys.intersection(cfg.keys()):
                return cfg
            if port_name in cfg or 'default' in cfg:
                return cfg.get(port_name, cfg.get('default'))
        return cfg

    def resolve_rebalance_config(port_dict):
        if 'rebalance' in port_dict:
            port_reb = port_dict.get('rebalance')
            if port_reb == 'no_rebalance':
                return None
            return port_reb
        if rebalance == 'no_rebalance':
            return None
        return rebalance
    # ---------------------------------------------------------------------
    # STEP 1: Collect all unique index names across all portfolios
    # ---------------------------------------------------------------------
    all_index_names = set()
    for port in portfolio_list:
        for idx_name, _ in port['weights']:
            all_index_names.add(idx_name)
    all_index_names = sorted(all_index_names)

    print("=" * 100)
    print(f" COMPARING {len(portfolio_list)} PORTFOLIOS")
    print("=" * 100)
    print(f" Unique indices needed: {len(all_index_names)}")
    for name in all_index_names:
        print(f"   - {name}")

    # ---------------------------------------------------------------------
    # STEP 2: Load all indices and find the common date range
    # ---------------------------------------------------------------------
    print("\n📂 Loading index data...")
    index_data = {}
    common_dates = None

    for idx_name in all_index_names:
        df = load_index_files(idx_name, search_path, recursive, data_format=data_format)
        if df is None or len(df) == 0:
            raise ValueError(f"Could not load data for {idx_name}")
        df = df.rename(columns={'Total Returns Index': idx_name})
        df = df.set_index('Date')
        index_data[idx_name] = df[idx_name]

        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    if len(common_dates) == 0:
        raise ValueError("No common dates found among the indices.")

    # Apply user's start/end date filters
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        common_dates = common_dates[common_dates >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        common_dates = common_dates[common_dates <= end_date]

    if len(common_dates) == 0:
        raise ValueError(f"No data available between {start_date} and {end_date}")

    common_dates = common_dates.sort_values()
    print(f"\n✅ Global common date grid: {common_dates.min().date()} to {common_dates.max().date()} ({len(common_dates)} days)")

    # ---------------------------------------------------------------------
    # STEP 3: Build each portfolio on the common date grid
    # ---------------------------------------------------------------------
    all_results = {}
    all_data = {}

    for idx, port in enumerate(portfolio_list, 1):
        name = port['name']
        weights = port['weights']
        print(f"\n[{idx}/{len(portfolio_list)}] Processing: {name}")
        print("-" * 60)

        try:
            # Build portfolio using the pre‑computed common grid
            port_reb_cfg = resolve_rebalance_config(port)
            if port_reb_cfg:
                port_tax_cfg = port.get('tax_config')
                if port_tax_cfg is None:
                    port_tax_cfg = resolve_tax_config(tax_config, name, idx - 1, len(portfolio_list))
                if port_tax_cfg is None:
                    raise ValueError(f"tax_config is required for rebalanced portfolio: {name}")
                need_dates = plot_rebalance_marks
                need_history = show_rebalance_history
                need_details = need_dates or need_history
                if need_details:
                    reb_result = build_rebalanced_portfolio_on_grid(
                        weights, index_data, common_dates, port_reb_cfg, port_tax_cfg, return_details=True
                    )
                    port_df = reb_result['data']
                else:
                    port_df = build_rebalanced_portfolio_on_grid(
                        weights, index_data, common_dates, port_reb_cfg, port_tax_cfg
                    )
            else:
                port_df = build_portfolio_series_on_grid(
                    weights, index_data, common_dates
                )

            # Analyze it
            results = analyze_portfolio(
                portfolio_name=name,
                portfolio_df=port_df,
                risk_free_rate=risk_free_rate,
                sip_amount=sip_amount,
                sip_day=sip_day,
                trading_days_per_year=trading_days_per_year,
                rolling_periods=rolling_periods,
                show_plots=show_plots,
                export_csv=export_csv,
                index_number=idx
            )


            if results and 'summary' in results:
                all_results[name] = results['summary']
                all_data[name] = {
                    'data': results.get('data'),
                    'sip_xirr': results.get('sip_xirr_series')
                }
                if port_reb_cfg and need_details:
                    rebalance_entries = reb_result.get('rebalance_log', [])
                    if need_dates:
                        rebalance_dates = [
                            e['date'] for e in rebalance_entries if e.get('event') == 'rebalance'
                        ]
                        all_data[name]['rebalance_dates'] = rebalance_dates
                    if need_history:
                        all_data[name]['rebalance_history'] = rebalance_entries
                print(f"✅ Successfully processed: {name}")
            else:
                print(f"⚠️  No results for: {name}")

        except Exception as e:
            print(f"❌ Error processing {name}: {str(e)}")
            continue

    if not all_results:
        print("\n❌ No portfolios could be processed.")
        return None

    # ---------------------------------------------------------------------
    # STEP 4: Combine summaries and display
    # ---------------------------------------------------------------------
    combined_df = pd.DataFrame(all_results).T

    # (Column ordering as before)
    # base_cols = [
    #     'Date Range', 'Total Trading Days', 'Start Value', 'End Value',
    #     'Total Return %', 'CAGR %', 'Annualized Volatility %', 'Sharpe Ratio',
    #     'Max Drawdown %', 'Max Drawdown Date', 'SIP XIRR %', 'SIP Return %',
    #     'SIP Invested Amt', 'SIP Final Amt',
    #     'Positive Days %', 'Negative Days %', 'Max Daily Return %', 'Min Daily Return %'
    # ]
    base_cols = [
        'Date Range', 
        'Total Return %', 'CAGR %', 'Annualized Volatility %', 'Sharpe Ratio',
        'Max Drawdown %', 'SIP XIRR %', 'SIP Return %',
        'SIP Invested Amt', 'SIP Final Amt', 'Start Value', 'End Value',
        'Positive Days %', 'Negative Days %', 'Max Daily Return %', 'Min Daily Return %'
    ]
    order = base_cols[:]
    for p in rolling_periods:
        order.extend([f'{p}Y Rolling Avg %', f'{p}Y Rolling Median %',
                      f'{p}Y Rolling Min %', f'{p}Y Rolling Max %'])
    for y in range(5):
        yr = datetime.now().year - y
        order.append(f'{yr} Return %')
    existing = [c for c in order if c in combined_df.columns]
    combined_df = combined_df[existing]

    print("\n📊 COMBINED SUMMARY - ALL PORTFOLIOS (COMMON DATES)")
    print("=" * 100)
    display_df = combined_df.copy()
    for col in display_df.columns:
        if any(x in col for x in ['Return', 'CAGR', 'Volatility', 'Drawdown']):
            if display_df[col].dtype in [np.float64, np.int64]:
                display_df[col] = display_df[col].round(2).astype(str) + '%'
        elif 'Invested Amt' in col or 'Final Amt' in col:
            if display_df[col].dtype in [np.float64, np.int64]:
                display_df[col] = display_df[col].round(2).astype(str)
        elif 'Sharpe' in col:
            display_df[col] = display_df[col].round(3)
        elif 'XIRR' in col:
            display_df[col] = display_df[col].round(2).astype(str) + '%'
    print(display_df.to_string())

    # After building all_data, add percentile summary if requested
    if display_rolling_returns_table and all_data:
        display_rolling_returns_summary(all_data, rolling_periods, rolling_percentiles_to_show, trading_days_per_year)

    # Generate comparison plots (they now automatically use the common grid)
    if len(all_results) > 0:
        print("\n📈 Generating comparison plots...")
        if plot_comparison:
            plot_portfolio_comparison(all_data, all_results, rolling_periods)
        if plot_running_xirr > 0 : 
            plot_running_xirr_comparison(all_data, all_results)
        if plot_running_cagr:
            if interactive_cagr:
                # Use interactive widget with all available periods
                plot_running_cagr_comparison_interactive(
                    all_data, all_results, trading_days_per_year,
                    show_percentiles=plot_rolling_returns_percentiles,
                    rolling_percentiles_to_show=rolling_percentiles_to_show,
                    available_periods=running_cagr_periods
                )
            else:
                # Original behavior: separate plots
                for period in running_cagr_periods:
                    plot_running_cagr_comparison(
                        all_data, all_results, period, trading_days_per_year,
                        show_percentiles=plot_rolling_returns_percentiles,
                        rolling_percentiles_to_show=rolling_percentiles_to_show
                    )

    if export_csv:
        fname = f'portfolios_comparison_summary_{output_fname}.csv' if output_fname else 'portfolios_comparison_summary.csv'
        combined_df.to_csv(fname)
        print(f"\n📁 Exported: {fname}")

    if show_rebalance_history:
        for name, data_dict in all_data.items():
            history = data_dict.get('rebalance_history')
            if not history:
                continue
            rows = []
            rebalance_dates = []
            for entry in history:
                if entry.get('event') != 'rebalance':
                    continue
                if entry.get('date') is not None:
                    rebalance_dates.append(entry['date'])
                rows.append({
                    'Portfolio': name,
                    'Date': entry.get('date').date() if entry.get('date') is not None else None,
                    'Weights Before': entry.get('weights_before'),
                    'Weights After': entry.get('weights_after'),
                    'Sells': entry.get('trades', {}).get('sells'),
                    'Buys': entry.get('trades', {}).get('buys'),
                    'ST Gain': entry.get('realized', {}).get('st_gain'),
                    'LT Gain': entry.get('realized', {}).get('lt_gain'),
                    'ST Loss': entry.get('realized', {}).get('st_loss'),
                    'LT Loss': entry.get('realized', {}).get('lt_loss')
                })
            if rows:
                print(f"\n📑 REBALANCE HISTORY: {name}")
                print(pd.DataFrame(rows).to_string(index=False))
                if len(rebalance_dates) > 1:
                    rebalance_dates = sorted(rebalance_dates)
                    gaps = [(rebalance_dates[i] - rebalance_dates[i - 1]).days for i in range(1, len(rebalance_dates))]
                    print(f"   ⏱️  Rebalance gaps (days): count={len(rebalance_dates)}, avg={np.mean(gaps):.1f}, median={np.median(gaps):.1f}, min={np.min(gaps)}, max={np.max(gaps)}")
                elif len(rebalance_dates) == 1:
                    print("   ⏱️  Rebalance gaps (days): count=1")

    return combined_df
# =============================================================================
# 6. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Define two portfolio options
    portfolios_to_compare = [
        {
            'name': 'Small+Mid+200Mom30',
            'weights': [
                ('NIFTY SMALLCAP250 MOMENTUM QUALITY 100', 1/3),
                ('NIFTY MIDCAP150 MOMENTUM 50', 1/3),
                ('NIFTY200 MOMENTUM 30', 1/3)
            ]
        },
        {
            'name': '400Mom100+200Mom30',
            'weights': [
                ('NIFTY400 MOMENTUM 100', 2/3),
                ('NIFTY200 MOMENTUM 30', 1/3)
            ]
        }
    ]

    # Run the comparison
    summary = compare_portfolios(
        portfolios_to_compare,
        risk_free_rate=6.0,
        sip_amount=10000,
        show_plots=True,
        export_csv=True,
        search_path='.',          # where your CSV files are
        recursive=True,
        output_fname="example"
    )

    # View the summary DataFrame
    if summary is not None:
        print("\n📊 FINAL SUMMARY")
        print(summary)
