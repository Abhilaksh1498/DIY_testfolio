#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profit booking utilities for composite portfolio analysis.
"""

import numpy as np
import pandas as pd
import bisect

from US_indices.rebalance import (
    _normalize_weights,
    _normalize_tax_config,
    _sell_units_fifo,
    _apply_tax_netting,
    _fy_year
)


def _validate_profit_booking_config(cfg):
    if cfg is None:
        return None
    method = cfg.get('method', 'cagr_percentile')
    if method != 'cagr_percentile':
        raise ValueError(f"Unsupported profit booking method: {method}")
    window_years = float(cfg.get('window_years', 3))
    percentile = float(cfg.get('percentile', 90))
    reentry_percentile = float(cfg.get('reentry_percentile', 25))
    trim_pct = float(cfg.get('trim_pct', 0.1))
    min_days_between = int(cfg.get('min_days_between', 252))
    reentry_min_days_between = int(cfg.get('reentry_min_days_between', 0))
    trading_days_per_year = int(cfg.get('trading_days_per_year', 252))
    percentile_mode = cfg.get('percentile_mode', 'full_history')

    if window_years <= 0:
        raise ValueError("window_years must be > 0")
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be between 0 and 100")
    if not (0 <= reentry_percentile <= 100):
        raise ValueError("reentry_percentile must be between 0 and 100")
    if not (0 < trim_pct < 1):
        raise ValueError("trim_pct must be between 0 and 1 (fraction)")
    if percentile_mode not in ['expanding', 'full_history']:
        raise ValueError("percentile_mode must be 'expanding' or 'full_history'")

    return {
        'method': method,
        'window_years': window_years,
        'percentile': percentile,
        'reentry_percentile': reentry_percentile,
        'trim_pct': trim_pct,
        'min_days_between': min_days_between,
        'reentry_min_days_between': reentry_min_days_between,
        'trading_days_per_year': trading_days_per_year,
        'percentile_mode': percentile_mode
    }


def _normalize_cash_config(cfg, trading_days_per_year):
    if cfg is None:
        return {
            'type': 'fixed_rate',
            'annual_rate': 0.0,
            'trading_days_per_year': trading_days_per_year
        }
    cash_type = cfg.get('type', 'fixed_rate')
    if cash_type == 'fixed_rate':
        return {
            'type': 'fixed_rate',
            'annual_rate': float(cfg.get('annual_rate', 0.0)),
            'trading_days_per_year': int(cfg.get('trading_days_per_year', trading_days_per_year))
        }
    if cash_type == 'series':
        series_name = cfg.get('series_name')
        if not series_name:
            raise ValueError("cash_config type='series' requires 'series_name'")
        return {
            'type': 'series',
            'series_name': series_name
        }
    raise ValueError(f"Unsupported cash_config type: {cash_type}")


def _compute_rolling_cagr(price_series, window_years, trading_days_per_year):
    window_days = int(window_years * trading_days_per_year)
    return (price_series / price_series.shift(window_days)) ** (1 / window_years) - 1


def _compute_full_history_percentiles(series):
    values = series.values.astype(float)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return pd.Series(np.full_like(values, np.nan, dtype=float), index=series.index)
    sorted_vals = np.sort(values[mask])
    ranks = np.searchsorted(sorted_vals, values[mask], side='right')
    percentiles = np.full_like(values, np.nan, dtype=float)
    percentiles[mask] = ranks / len(sorted_vals) * 100
    return pd.Series(percentiles, index=series.index)


def _compute_expanding_percentiles(series):
    values = series.values.astype(float)
    percentiles = np.full_like(values, np.nan, dtype=float)
    sorted_vals = []
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        bisect.insort(sorted_vals, v)
        rank = bisect.bisect_right(sorted_vals, v)
        percentiles[i] = rank / len(sorted_vals) * 100
    return pd.Series(percentiles, index=series.index)


def build_profit_booking_portfolio_on_grid(portfolio_def, index_data, common_dates,
                                           profit_booking_cfg, cash_cfg=None,
                                           tax_cfg=None, return_details=False):
    """
    Build a portfolio series with profit booking based on trailing CAGR percentiles.

    Parameters:
    -----------
    portfolio_def : list of (index_name, weight)
    index_data : dict of pd.Series, indexed by Date
    common_dates : iterable of datetime-like
    profit_booking_cfg : dict, must include method='cagr_percentile'
    cash_cfg : dict, fixed_rate or series
    tax_cfg : dict or str, same format as rebalance tax_config
    return_details : bool, include profit booking history
    """
    profit_booking_cfg = _validate_profit_booking_config(profit_booking_cfg)
    tax_cfg = _normalize_tax_config(tax_cfg)
    cash_cfg = _normalize_cash_config(cash_cfg, profit_booking_cfg['trading_days_per_year'])

    names, weights, target_weights, weight_sum = _normalize_weights(portfolio_def)
    date_index = pd.Index(common_dates).sort_values()
    price_df = pd.DataFrame({name: index_data[name].loc[date_index] for name in names})

    first_date = date_index[0]
    initial_total = weight_sum

    units = {}
    lots = {}
    for idx, name in enumerate(names):
        price0 = price_df.loc[first_date, name]
        alloc_value = initial_total * target_weights[idx]
        units[name] = alloc_value / price0
        lots[name] = [{
            'units': units[name],
            'cost_per_unit': price0,
            'date': first_date
        }]

    cash_value = 0.0
    last_book_date = {name: None for name in names}
    last_reentry_date = {name: None for name in names}
    booked_cash_by_asset = {name: 0.0 for name in names}
    realized = {'st_gain': 0.0, 'st_loss': 0.0, 'lt_gain': 0.0, 'lt_loss': 0.0}
    carry_forward = {'stcl': [], 'ltcl': []}
    pending_tax = 0.0
    total_tax_paid = 0.0
    last_date = None

    cash_returns = None
    if cash_cfg['type'] == 'series':
        cash_series = index_data.get(cash_cfg['series_name'])
        if cash_series is None:
            raise ValueError(f"cash series not found: {cash_cfg['series_name']}")
        cash_series = cash_series.reindex(date_index).ffill().bfill()
        cash_returns = cash_series.pct_change().fillna(0.0)

    rolling_cagr = {
        name: _compute_rolling_cagr(
            price_df[name],
            profit_booking_cfg['window_years'],
            profit_booking_cfg['trading_days_per_year']
        )
        for name in names
    }
    if profit_booking_cfg['percentile_mode'] == 'expanding':
        percentile_series = {name: _compute_expanding_percentiles(rolling_cagr[name]) for name in names}
    else:
        percentile_series = {name: _compute_full_history_percentiles(rolling_cagr[name]) for name in names}

    portfolio_values = []
    booking_log = []

    for date in date_index:
        # Apply cash returns
        if cash_value > 0:
            if cash_cfg['type'] == 'fixed_rate':
                daily_rate = (1 + cash_cfg['annual_rate']) ** (1 / cash_cfg['trading_days_per_year']) - 1
                cash_value *= (1 + daily_rate)
            else:
                cash_value *= (1 + cash_returns.loc[date])

        # Pay prior FY tax on first trading day of April
        if last_date is not None and _fy_year(date) != _fy_year(last_date):
            tax_due, carry_forward = _apply_tax_netting(
                realized['st_gain'], realized['st_loss'],
                realized['lt_gain'], realized['lt_loss'],
                tax_cfg, carry_forward
            )
            realized = {'st_gain': 0.0, 'st_loss': 0.0, 'lt_gain': 0.0, 'lt_loss': 0.0}
            pending_tax = tax_due

        if pending_tax > 0:
            if cash_value + 1e-12 < pending_tax:
                shortfall = pending_tax - cash_value
                prices = price_df.loc[date]
                asset_values = {name: units[name] * prices[name] for name in names}
                total_value = sum(asset_values.values())
                if total_value > 0:
                    for name in names:
                        sell_value = shortfall * (asset_values[name] / total_value)
                        price = prices[name]
                        units_to_sell = min(units[name], sell_value / price)
                        if units_to_sell > 1e-12:
                            result = _sell_units_fifo(
                                lots[name], units_to_sell, price, date, tax_cfg
                            )
                            units[name] -= result['units_sold']
                            lots[name] = result['lots']
                            cash_value += result['gross_proceeds']
                            realized['st_gain'] += result['st_gain']
                            realized['st_loss'] += result['st_loss']
                            realized['lt_gain'] += result['lt_gain']
                            realized['lt_loss'] += result['lt_loss']

            if cash_value + 1e-12 < pending_tax:
                pending_tax = cash_value

            cash_value -= pending_tax
            total_tax_paid += pending_tax
            if return_details:
                booking_log.append({
                    'date': date,
                    'event': 'tax_payment',
                    'tax_paid': pending_tax,
                    'cash_after': cash_value
                })
            pending_tax = 0.0

        prices = price_df.loc[date]
        asset_values = {name: units[name] * prices[name] for name in names}
        total_value = cash_value + sum(asset_values.values())

        # Profit booking per asset
        for name in names:
            cagr_today = rolling_cagr[name].loc[date]
            if pd.isna(cagr_today):
                continue

            percentile = percentile_series[name].loc[date]
            last_date_for_asset = last_book_date[name]
            if percentile >= profit_booking_cfg['percentile']:
                if last_date_for_asset is None or (date - last_date_for_asset).days >= profit_booking_cfg['min_days_between']:
                    units_to_sell = units[name] * profit_booking_cfg['trim_pct']
                    if units_to_sell > 1e-12:
                        result = _sell_units_fifo(lots[name], units_to_sell, prices[name], date, tax_cfg)
                        units[name] -= result['units_sold']
                        lots[name] = result['lots']
                        cash_value += result['gross_proceeds']
                        booked_cash_by_asset[name] += result['gross_proceeds']
                        realized['st_gain'] += result['st_gain']
                        realized['st_loss'] += result['st_loss']
                        realized['lt_gain'] += result['lt_gain']
                        realized['lt_loss'] += result['lt_loss']
                        last_book_date[name] = date
                        if return_details:
                            booking_log.append({
                                'date': date,
                                'event': 'profit_booking',
                                'asset': name,
                                'percentile': percentile,
                                'cagr': cagr_today,
                                'units_sold': result['units_sold'],
                                'value': result['gross_proceeds'],
                                'cash_after': cash_value,
                                'st_gain': result['st_gain'],
                                'lt_gain': result['lt_gain'],
                                'st_loss': result['st_loss'],
                                'lt_loss': result['lt_loss']
                            })

            reentry_last = last_reentry_date[name]
            if percentile <= profit_booking_cfg['reentry_percentile'] and booked_cash_by_asset[name] > 1e-12:
                if reentry_last is None or (date - reentry_last).days >= profit_booking_cfg['reentry_min_days_between']:
                    buy_value = min(booked_cash_by_asset[name], cash_value)
                    if buy_value > 1e-12:
                        units_to_buy = buy_value / prices[name]
                        units[name] += units_to_buy
                        cash_value -= buy_value
                        booked_cash_by_asset[name] -= buy_value
                        lots[name].append({
                            'units': units_to_buy,
                            'cost_per_unit': prices[name],
                            'date': date
                        })
                        last_reentry_date[name] = date
                        if return_details:
                            booking_log.append({
                                'date': date,
                                'event': 'reentry',
                                'asset': name,
                                'percentile': percentile,
                                'cagr': cagr_today,
                                'units_bought': units_to_buy,
                                'value': buy_value,
                                'cash_after': cash_value
                            })

        asset_values = {name: units[name] * prices[name] for name in names}
        total_value = cash_value + sum(asset_values.values())
        portfolio_values.append({
            'Date': date,
            'Portfolio_Value': total_value
        })
        last_date = date

    result_df = pd.DataFrame(portfolio_values).reset_index(drop=True)
    if return_details:
        return {
            'data': result_df,
            'profit_booking_log': booking_log,
            'total_tax_paid': total_tax_paid
        }
    return result_df
