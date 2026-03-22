# Cell: Multi-Index Analysis Function
"""
## Multi-Index Analysis Function
Analyze multiple indices and return a combined summary DataFrame for comparison.

**Usage:**
    indices = [
        "NIFTY SMALLCAP250 MOMENTUM QUALITY 100",
        "NIFTY500 MOMENTUM 50",
        "NIFTY MIDCAP150 MOMENTUM 50",
        "NIFTY SMALLCAP250"
    ]
    
    summary_df = analyze_indices(indices)
    summary_df.to_csv('indices_comparison.csv')
    display(summary_df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_indices(index_names, 
                    risk_free_rate=6.0,
                    sip_amount=10000,
                    sip_day=2,
                    trading_days_per_year=252,
                    rolling_periods=[3, 5, 7, 10],
                    show_plots=True,
                    export_csv=False,
                    plot_comparison=True,
                    search_path='.',           # NEW
                    recursive=True,
                    output_fname = ""):           
    """
    Analyze multiple indices and return a combined summary DataFrame.
    
    Parameters:
    -----------
    index_names : list
        List of index name prefixes (e.g., ["NIFTY SMALLCAP250 MOMENTUM QUALITY 100", "NIFTY500 MOMENTUM 50"])
    
    risk_free_rate : float, default=6.0
        Annual risk-free rate for Sharpe ratio calculation (%)
    
    sip_amount : float, default=10000
        Monthly SIP investment amount (₹)
    
    sip_day : int, default=2
        Day of month for SIP investment
    
    trading_days_per_year : int, default=252
        Number of trading days in a year
    
    rolling_periods : list, default=[3,5,7,10]
        Rolling return periods in years
    
    show_plots : bool, default=True
        Whether to display individual index plots
    
    export_csv : bool, default=False
        Whether to export individual index results to CSV files
    
    plot_comparison : bool, default=True
        Whether to generate comparison plots across indices
    
    Returns:
    --------
    pandas.DataFrame: Combined summary for all indices with index names as columns
    """
    
    # Initialize results dictionary
    all_results = {}
    all_data = {}  # Store full data for comparison plots
    
    print("=" * 100)
    print(f" MULTI-INDEX ANALYSIS: {len(index_names)} INDICES")
    print("=" * 100)
    
    for idx, index_name in enumerate(index_names, 1):
        print(f"\n[{idx}/{len(index_names)}] Processing: {index_name}")
        print("-" * 60)
        
        try:
            # Analyze single index
            results = analyze_single_index(
                index_name=index_name,
                risk_free_rate=risk_free_rate,
                sip_amount=sip_amount,
                sip_day=sip_day,
                trading_days_per_year=trading_days_per_year,
                rolling_periods=rolling_periods,
                show_plots=show_plots,
                export_csv=export_csv,
                index_number=idx,
                search_path=search_path,    # Pass through
                recursive=recursive,          # Pass through
            )
            
            if results and 'summary' in results:
                all_results[index_name] = results['summary']
                # all_data[index_name] = results.get('data', None)
                all_data[index_name] = {
                    'data': results.get('data'),
                    'sip_xirr': results.get('sip_xirr_series')      # NEW
                }
                print(f"✅ Successfully processed: {index_name}")
            else:
                print(f"⚠️ No results generated for: {index_name}")
                
        except Exception as e:
            print(f"❌ Error processing {index_name}: {str(e)}")
            continue
    
    if not all_results:
        print("\n❌ No indices were successfully processed!")
        return None
    
    # Create combined summary DataFrame
    print("\n" + "=" * 100)
    print(" CREATING COMBINED SUMMARY DATAFRAME")
    print("=" * 100)
    
    # Convert results to DataFrame
    combined_df = pd.DataFrame(all_results).T
    
    # Reorder columns for better readability
    column_order = [
        'Date Range',
        'Total Trading Days',
        'Start Value',
        'End Value',
        'Total Return %',
        'CAGR %',
        'Annualized Volatility %',
        'Sharpe Ratio',
        'Max Drawdown %',
        'Max Drawdown Date',
        'SIP XIRR %',
        'SIP Return %',
        'Positive Days %',
        'Negative Days %',
        'Max Daily Return %',
        'Min Daily Return %'
    ]
    
    # Add rolling return columns
    for period in rolling_periods:
        column_order.append(f'{period}Y Rolling Avg %')
        column_order.append(f'{period}Y Rolling Median %')
        column_order.append(f'{period}Y Rolling Min %')
        column_order.append(f'{period}Y Rolling Max %')
    
    # Add yearly return columns (last 5 years)
    for year_offset in range(5):
        year = datetime.now().year - year_offset
        column_order.append(f'{year} Return %')
    
    # Ensure all columns exist
    existing_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[existing_columns]
    
    # Display combined summary
    print("\n📊 COMBINED SUMMARY - ALL INDICES")
    print("=" * 100)
    
    # Format for display
    display_df = combined_df.copy()
    for col in display_df.columns:
        if 'Return' in col or 'CAGR' in col or 'Volatility' in col or 'Drawdown' in col:
            if display_df[col].dtype in [np.float64, np.int64]:
                display_df[col] = display_df[col].round(2).astype(str) + '%'
        elif 'Sharpe' in col:
            display_df[col] = display_df[col].round(3)
        elif 'XIRR' in col:
            display_df[col] = display_df[col].round(2).astype(str) + '%'
    
    print(display_df.to_string())
    
    # Generate comparison plots
    if plot_comparison and len(all_results) > 1:
        plot_index_comparison(all_data, all_results, rolling_periods)
        plot_running_xirr_comparison(all_data, all_results)   # NEW
    
    # Export combined summary if requested
    if export_csv:
        if (output_fname == ""):
            combined_df.to_csv('indices_comparison_summary.csv')
        else:
            combined_df.to_csv(f'indices_comparison_summary_{output_fname}.csv')
        print("\n📁 Exported: indices_comparison_summary.csv")
    
    print("\n" + "=" * 100)
    print("✅ MULTI-INDEX ANALYSIS COMPLETE")
    print("=" * 100)
    
    return combined_df


def analyze_single_index(index_name, 
                         risk_free_rate=6.0,
                         sip_amount=10000,
                         sip_day=2,
                         trading_days_per_year=252,
                         rolling_periods=[3, 5, 7, 10],
                         show_plots=True,
                         export_csv=False,
                         index_number=1,
                    search_path='.',           # NEW
                    recursive=True):           # NEW):
    """
    Analyze a single index and return summary statistics.
    This is an internal function called by analyze_indices().
    """
    
    # =========================================================================
    # STEP 1: FIND AND LOAD FILES
    # =========================================================================
    
    def parse_date(date_str):
        """Parse date strings in multiple formats"""
        formats = ['%d %b %Y', '%d-%b-%Y', '%Y-%m-%d']
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        return None
    
    # def load_index_files(file_prefix):
    #     """Load all CSV files matching the prefix"""
        
    #     # Try different patterns
    #     patterns = [
    #         f"{file_prefix}_Historical_TR_*.csv",
    #         f"{file_prefix}*.csv",
    #         f"*{file_prefix}*.csv"
    #     ]
        
    #     all_files = []
    #     for pattern in patterns:
    #         all_files.extend(glob.glob(pattern))
        
    #     all_files = list(set(all_files))  # Remove duplicates
        
    #     if not all_files:
    #         print(f"   ⚠️ No files found for: {file_prefix}")
    #         return None
        
    #     all_data = []
        
    #     for file_path in sorted(all_files):
    #         try:
    #             df = pd.read_csv(file_path)
    #             df.columns = df.columns.str.strip()
                
    #             # Identify index value column
    #             value_col = None
    #             for col in df.columns:
    #                 if 'total returns' in col.lower() or 'index value' in col.lower() or 'close' in col.lower():
    #                     value_col = col
    #                     break
                
    #             if value_col is None:
    #                 value_col = df.columns[-1]
                
    #             # Rename for consistency
    #             df = df.rename(columns={df.columns[0]: 'Date', value_col: 'Index_Value'})
                
    #             # Parse dates
    #             df['Date'] = df['Date'].apply(parse_date)
    #             df = df.dropna(subset=['Date'])
    #             df = df[['Date', 'Index_Value']].copy()
    #             df = df.sort_values('Date').drop_duplicates('Date')
                
    #             all_data.append(df)
                
    #         except Exception as e:
    #             print(f"   ⚠️ Error loading {os.path.basename(file_path)}: {str(e)}")
    #             continue
        
    #     if not all_data:
    #         return None
        
    #     combined = pd.concat(all_data, ignore_index=True)
    #     combined = combined.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
        
    #     return combined

    # Calculate XIRR (simplified approximation)
    def calculate_xirr(flows, dates, guess=0.1):
        """Calculate XIRR using Newton-Raphson method"""
        if len(flows) != len(dates):
            return None
        
        days = [(d - dates[0]).days for d in dates]
        
        def npv(rate):
            return sum(f / (1 + rate) ** (d/365) for f, d in zip(flows, days))
        
        rate = guess
        for _ in range(100):
            f = npv(rate)
            f_prime = sum(-f * d / (365 * (1 + rate) ** (d/365 + 1)) 
                        for f, d in zip(flows, days))
            
            if abs(f_prime) < 1e-10:
                break
            
            new_rate = rate - f / f_prime
            if abs(new_rate - rate) < 1e-8:
                return new_rate
            rate = new_rate
        
        return rate
 
    def load_index_files(file_prefix, search_path='.', recursive=True):
        """Load all CSV files matching the prefix using recursive glob"""
        
        import glob
        import os
        
        # Construct pattern: any CSV containing the prefix anywhere in the filename
        if recursive:
            pattern = os.path.join(search_path, '**', f'*{file_prefix}*.csv')
        else:
            pattern = os.path.join(search_path, f'*{file_prefix}*.csv')
        
        all_files = glob.glob(pattern, recursive=recursive)
        
        if not all_files:
            print(f"   ⚠️ No files found matching pattern: {pattern}")
            return None
        
        print(f"   Found {len(all_files)} files:")
        for f in all_files:
            print(f"      - {os.path.basename(f)}")
        
        all_data = []
        for file_path in sorted(all_files):
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                
                # Identify index value column
                value_col = None
                for col in df.columns:
                    if 'total returns' in col.lower() or 'index value' in col.lower() or 'close' in col.lower():
                        value_col = col
                        break
                
                if value_col is None:
                    value_col = df.columns[-1]  # fallback
                
                # df = df.rename(columns={df.columns[0]: 'Date', value_col: 'Index_Value'})
                
                # Parse dates
                df['Date'] = df['Date'].apply(parse_date)
                df = df.dropna(subset=['Date'])
                # df = df[['Date', 'Total Returns Index']].copy()
                df = df.sort_values('Date').drop_duplicates('Date')
                
                all_data.append(df)
                
            except Exception as e:
                print(f"   ⚠️ Error loading {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_data:
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
        return combined
    
    # Load the data with new parameters
    df = load_index_files(index_name, search_path=search_path, recursive=recursive)
    
    if df is None or len(df) == 0:
        print(f"   ❌ No data loaded for: {index_name}")
        return None
    
    print(f"   ✓ Data loaded: {df['Date'].min().date()} to {df['Date'].max().date()} ({len(df):,} days)")
    
    # =========================================================================
    # STEP 2: CALCULATE RETURNS
    # =========================================================================
    
    df['Daily_Return'] = df['Total Returns Index'].pct_change() * 100
    df = df.dropna().reset_index(drop=True)
    
    df['Cumulative_Return'] = (1 + df['Daily_Return']/100).cumprod()
    df['Running_Max'] = df['Total Returns Index'].cummax()
    df['Drawdown'] = (df['Total Returns Index'] / df['Running_Max'] - 1) * 100
    
    # =========================================================================
    # STEP 3: BASIC STATISTICS
    # =========================================================================
    
    start_val = df['Total Returns Index'].iloc[0]
    end_val = df['Total Returns Index'].iloc[-1]
    total_days = len(df)
    years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    
    total_return_pct = (end_val / start_val - 1) * 100
    cagr = ((end_val / start_val) ** (1/years) - 1) * 100
    
    daily_returns = df['Daily_Return']
    
    # =========================================================================
    # STEP 4: SHARPE RATIO
    # =========================================================================
    
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days_per_year)
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0
    
    # =========================================================================
    # STEP 5: ROLLING RETURNS
    # =========================================================================
    
    rolling_stats = {}
    
    for period in rolling_periods:
        rolling_days = int(period * trading_days_per_year)
        
        if len(df) > rolling_days:
            rolling_returns = []
            
            for i in range(rolling_days, len(df)):
                start = df.loc[i - rolling_days, 'Total Returns Index']
                end = df.loc[i, 'Total Returns Index']
                ann_return = ((end / start) ** (1/period) - 1) * 100
                rolling_returns.append(ann_return)
            
            if rolling_returns:
                rolling_stats[f'{period}Y'] = {
                    'avg': np.mean(rolling_returns),
                    'median': np.median(rolling_returns),
                    'min': np.min(rolling_returns),
                    'max': np.max(rolling_returns),
                    'std': np.std(rolling_returns)
                }
    
    # =========================================================================
    # STEP 6: DRAWDOWN
    # =========================================================================
    
    max_dd = df['Drawdown'].min()
    max_dd_date = df.loc[df['Drawdown'].idxmin(), 'Date'].date()
    
    # =========================================================================
    # STEP 7: SIP RETURNS
    # =========================================================================
    
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
                'Total Returns Index': closest['Total Returns Index']
            })
        
        sip_df = pd.DataFrame(sip_dates).sort_values('Date').reset_index(drop=True)
        
        total_investment = 0
        units = 0
        running_xirr = []      # XIRR values at each installment date
        running_dates = []     # corresponding dates
        
        for i, row in sip_df.iterrows():
            units += amount / row['Total Returns Index']
            total_investment += amount
            current_value = units * row['Total Returns Index']

            # Build cash flows up to now: i+1 outflows, one inflow (current value)
            flows = [-amount] * (i+1) + [current_value]
            dates = sip_df['Date'].iloc[:i+1].tolist() + [row['Date']]
            
            xirr = calculate_xirr(flows, dates)
            if xirr is not None and np.isreal(xirr):
    # Only store values within a reasonable range (-99% to +1000%)
                if -0.99 <= xirr <= 10:
                    running_xirr.append(xirr * 100)
                    running_dates.append(row['Date'])
                else:
                    # Optionally skip or append NaN
                    pass  # skip this point
        
        final_value = units * sip_df['Total Returns Index'].iloc[-1]
        return_pct = (final_value / total_investment - 1) * 100
        final_xirr = running_xirr[-1] if running_xirr else None
        
        # Simple XIRR approximation
        # years_sip = (sip_df['Date'].iloc[-1] - sip_df['Date'].iloc[0]).days / 365.25
        # xirr = ((final_value / total_investment) ** (1/years_sip) - 1) * 100 if years_sip > 0 else 0

        # Prepare cash flows for XIRR
        cash_flows = [-sip_amount] * (len(sip_df) - 1) + [final_value]
        xirr = calculate_xirr(cash_flows, sip_df['Date'].tolist())
        xirr_pct = xirr * 100 if xirr else None
        
        return {
            'return_pct': return_pct,
            'xirr': round(xirr_pct, 2) if xirr_pct else None,
            'installments': len(sip_df),
            'sip_start': sip_df['Date'].iloc[0].date(),
            'sip_end': sip_df['Date'].iloc[-1].date(),
            'xirr_series': pd.Series(running_xirr, index=running_dates)   # NEW
        }
    
    sip_results = calculate_sip_returns(df, sip_amount, sip_day)
    
    # =========================================================================
    # STEP 8: YEARLY RETURNS
    # =========================================================================
    
    df['Year'] = df['Date'].dt.year
    yearly_returns = {}
    
    for year in sorted(df['Year'].unique(), reverse=True)[:5]:  # Last 5 years
        year_df = df[df['Year'] == year]
        if len(year_df) > 0:
            year_return = (year_df['Total Returns Index'].iloc[-1] / year_df['Total Returns Index'].iloc[0] - 1) * 100
            yearly_returns[f'{year} Return %'] = year_return
    
    # =========================================================================
    # STEP 9: CREATE SUMMARY DICTIONARY
    # =========================================================================
    
    summary = {
        'Index': index_name,
        'Date Range': f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        'Total Trading Days': len(df),
        'Start Value': round(start_val, 2),
        'End Value': round(end_val, 2),
        'Total Return %': round(total_return_pct, 2),
        'CAGR %': round(cagr, 2),
        'Annualized Volatility %': round(annual_vol, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown %': round(max_dd, 2),
        'Max Drawdown Date': max_dd_date,
        'SIP XIRR %': round(sip_results['xirr'], 2),
        'SIP Return %': round(sip_results['return_pct'], 2),
        'Positive Days %': round((daily_returns > 0).mean() * 100, 2),
        'Negative Days %': round((daily_returns < 0).mean() * 100, 2),
        'Max Daily Return %': round(daily_returns.max(), 2),
        'Min Daily Return %': round(daily_returns.min(), 2)
    }
    
    # Add rolling return stats
    for period in rolling_periods:
        if f'{period}Y' in rolling_stats:
            stats = rolling_stats[f'{period}Y']
            summary[f'{period}Y Rolling Avg %'] = round(stats['avg'], 2)
            summary[f'{period}Y Rolling Median %'] = round(stats['median'], 2)
            summary[f'{period}Y Rolling Min %'] = round(stats['min'], 2)
            summary[f'{period}Y Rolling Max %'] = round(stats['max'], 2)
    
    # Add yearly returns
    summary.update(yearly_returns)
    
    # Optional: Generate plots for individual index
    if show_plots:
        plot_single_index(df, index_name, index_number, rolling_periods, rolling_stats)
    
    # Optional: Export individual results
    if export_csv:
        df.to_csv(f'{index_name.replace(" ", "_")}_data.csv', index=False)
        print(f"   📁 Exported: {index_name.replace(' ', '_')}_data.csv")
    
    return {
        'summary': summary,
        'data': df,
        'rolling_stats': rolling_stats,
        'sip_xirr_series': sip_results.get('xirr_series')   # NEW
    }


def plot_single_index(df, index_name, index_number, rolling_periods, rolling_stats):
    """Generate plots for a single index"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{index_number}. {index_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Index Value
    axes[0, 0].plot(df['Date'], df['Total Returns Index'], linewidth=1.5, color='#1f77b4')
    axes[0, 0].set_title('Index Value', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Returns Distribution
    axes[0, 1].hist(df['Daily_Return'], bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
    axes[0, 1].axvline(df['Daily_Return'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {df['Daily_Return'].mean():.3f}%")
    axes[0, 1].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Returns
    axes[1, 0].plot(df['Date'], df['Cumulative_Return'], linewidth=1.5, color='#ff7f0e')
    axes[1, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Cumulative Return (x times)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
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


def plot_index_comparison(all_data, all_results, rolling_periods):
    """Generate comparison plots across multiple indices"""
    
    num_indices = len(all_data)
    if num_indices < 2:
        return
    
    # Create color map for consistent colors
    colors = plt.cm.Set1(np.linspace(0, 1, num_indices))
    
    # Plot 1: Normalized performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Index Comparison Dashboard', fontsize=16, fontweight='bold')
    
    # Normalized performance (starting at 100)
    ax1 = axes[0, 0]
    for idx, (name, df) in enumerate(all_data.items()):
        df = df['data']
        if df is not None and len(df) > 0:
            normalized = df['Total Returns Index'] / df['Total Returns Index'].iloc[0] * 100
            ax1.plot(df['Date'], normalized, linewidth=1.5, color=colors[idx], label=name[:30])
    
    ax1.set_title('Normalized Performance (Start = 100)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Value')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # CAGR Comparison
    ax2 = axes[0, 1]
    names = list(all_results.keys())
    cagrs = [all_results[name]['CAGR %'] for name in names]
    colors_bars = ['green' if x > 0 else 'red' for x in cagrs]
    bars = ax2.bar(range(len(names)), cagrs, color=colors_bars, alpha=0.7)
    ax2.set_title('CAGR Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('CAGR (%)')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cagrs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Sharpe Ratio Comparison
    ax3 = axes[1, 0]
    sharpes = [all_results[name]['Sharpe Ratio'] for name in names]
    bars = ax3.bar(range(len(names)), sharpes, color='purple', alpha=0.7)
    ax3.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Max Drawdown Comparison
    ax4 = axes[1, 1]
    drawdowns = [abs(all_results[name]['Max Drawdown %']) for name in names]
    bars = ax4.bar(range(len(names)), drawdowns, color='red', alpha=0.7)
    ax4.set_title('Max Drawdown Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, drawdowns)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Rolling Returns Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    rolling_data = []
    for name in names:
        row = [name[:20]]
        for period in rolling_periods:
            col_name = f'{period}Y Rolling Median %'
            if col_name in all_results[name]:
                row.append(all_results[name][col_name])
            else:
                row.append(np.nan)
        rolling_data.append(row)
    
    columns = ['Index'] + [f'{p}Y Median' for p in rolling_periods]
    rolling_df = pd.DataFrame(rolling_data, columns=columns)
    rolling_df.set_index('Index', inplace=True)
    
    sns.heatmap(rolling_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                linewidths=0.5, cbar_kws={'label': 'Rolling Return (%)'})
    ax.set_title('Rolling Returns Comparison (Median %)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_running_xirr_comparison(all_data, all_results):
    """Plot running XIRR for all indices over time."""
    if not all_data:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))
    
    for idx, (name, data_dict) in enumerate(all_data.items()):
        xirr_series = data_dict.get('sip_xirr')
        xirr_series = xirr_series[xirr_series.index > pd.Timestamp('2016-01-01')]  # Filter out any invalid dates
        # print(f"######### {xirr_series} ############")
        if xirr_series is not None and len(xirr_series) > 0:
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
# =============================================================================
# QUICK ACCESS FUNCTIONS FOR COMMONLY USED INDICES
# =============================================================================

def compare_momentum_indices():
    """Quick comparison of popular momentum indices"""
    
    indices = [
        "NIFTY SMALLCAP250 MOMENTUM QUALITY 100",
        "NIFTY500 MOMENTUM 50",
        "NIFTY MIDCAP150 MOMENTUM 50",
        "NIFTY200 MOMENTUM 30"
    ]
    
    return analyze_indices(indices)


def compare_cap_based_indices():
    """Compare large, mid, small cap indices"""
    
    indices = [
        "NIFTY 50",
        "NIFTY MIDCAP 100",
        "NIFTY SMALLCAP 250",
        "NIFTY500 MOMENTUM 50"
    ]
    
    return analyze_indices(indices)


def compare_factor_indices():
    """Compare different factor-based indices"""
    
    indices = [
        "NIFTY500 MOMENTUM 50",
        "NIFTY500 VALUE 50",
        "NIFTY LOW VOLATILITY 50",
        "NIFTY200 QUALITY 30"
    ]
    
    return analyze_indices(indices)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Analyze a single index
    # result = analyze_indices(["NIFTY SMALLCAP250 MOMENTUM QUALITY 100"])
    
    # Example 2: Compare multiple indices
    indices_to_compare = [
        # "NIFTY500 VALUE 50",
        # "NIFTY500 MULTICAP MOMENTUM QUALITY 50",
        "NIFTY500 MOMENTUM 50",
        "NIFTY200 MOMENTUM 30",
        # "NIFTY TOTAL MARKET MOMENTUM QUALITY 50",
        # "NIFTY MIDSMALLCAP400 MOMENTUM QUALITY 100",
        "NIFTY MIDCAP150 MOMENTUM 50",
        "NIFTY SMALLCAP250 MOMENTUM QUALITY 100",
    ]
    
    summary_df = analyze_indices(
        index_names=indices_to_compare,
        risk_free_rate=6.0,
        sip_amount=10000,
        show_plots=False,
        export_csv=True,
        plot_comparison=True,
        search_path='.',           # current directory
        recursive=True,             # include subfolders
        output_fname="buy_and_hold"
    )
    
    # Display the summary DataFrame
    print("\n📊 FINAL COMPARISON SUMMARY")
    print(summary_df)