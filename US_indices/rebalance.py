#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebalancing utilities for composite portfolio analysis.
"""

import pandas as pd
import numpy as np


TAX_PROFILES = {
    'india_equity': {
        'stcg_rate': 0.20,
        'ltcg_rate': 0.125,
        'stcg_days': 0,
        'ltcg_days': 365,
        'ltcg_exemption': 125000.0,
        'apply_ltcg_exemption': True
    },
    'offshore_equity': {
        'stcg_rate': 0.30,
        'ltcg_rate': 0.125,
        'stcg_days': 0,
        'ltcg_days': 730,
        'ltcg_exemption': 0.0,
        'apply_ltcg_exemption': False
    },
    'none': {
        'stcg_rate': 0.0,
        'ltcg_rate': 0.0,
        'stcg_days': 0,
        'ltcg_days': 365,
        'ltcg_exemption': 0.0,
        'apply_ltcg_exemption': False
    }
}


def _normalize_weights(portfolio_def):
    names = [name for name, _ in portfolio_def]
    weights = np.array([w for _, w in portfolio_def], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Total weight must be positive.")
    target_weights = weights / total
    return names, weights, target_weights, total


def _validate_rebalance_config(rebalance_cfg):
    if rebalance_cfg is None:
        return None
    method = rebalance_cfg.get('method', 'threshold')
    if method == 'threshold':
        threshold = rebalance_cfg.get('drift_threshold', rebalance_cfg.get('threshold'))
        if threshold is None:
            raise ValueError("rebalance_cfg requires 'drift_threshold' for threshold method.")
        if threshold < 0:
            raise ValueError("drift_threshold must be >= 0.")
        min_days = int(rebalance_cfg.get('min_days_between', 0))
        return {
            'method': method,
            'drift_threshold': float(threshold),
            'min_days_between': min_days
        }
    if method == 'interval':
        interval = rebalance_cfg.get('interval')
        interval_months = rebalance_cfg.get('interval_months')
        interval_years = rebalance_cfg.get('interval_years')
        if interval is None and interval_months is None and interval_years is None:
            raise ValueError("interval method requires 'interval' or 'interval_months' or 'interval_years'")
        return {
            'method': method,
            'interval': interval,
            'interval_months': interval_months,
            'interval_years': interval_years,
            'min_days_between': int(rebalance_cfg.get('min_days_between', 0))
        }
    raise ValueError(f"Unsupported rebalance method: {method}")


def _compute_interval_rebalance_dates(date_index, cfg):
    dates = pd.Index(date_index).sort_values()
    if len(dates) == 0:
        return set()

    interval = (cfg.get('interval') or '').lower() if cfg.get('interval') else None
    interval_months = cfg.get('interval_months')
    interval_years = cfg.get('interval_years')

    if interval in ['quarterly', 'quarter']:
        interval_months = 3
    elif interval in ['biannual', 'bi-annually', 'semiannual', 'semi-annual', 'half-yearly', 'half yearly']:
        interval_months = 6
    elif interval in ['annual', 'annually', 'yearly', 'year']:
        interval_months = 12

    date_series = dates.to_series()
    is_first_of_month = date_series.dt.to_period('M') != date_series.dt.to_period('M').shift(1)

    if interval_years is not None:
        interval_years = int(interval_years)
        base_year = dates[0].year
        eligible = (date_series.dt.year - base_year) % interval_years == 0
        is_first_of_year = (date_series.dt.month == 1) & is_first_of_month
        rebalance_mask = eligible & is_first_of_year
    else:
        interval_months = int(interval_months) if interval_months is not None else 12
        if interval_months <= 0:
            return set()
        month_mod = (date_series.dt.month - 1) % interval_months == 0
        rebalance_mask = is_first_of_month & month_mod

    rebalance_dates = set(dates[rebalance_mask].tolist())
    if dates[0] in rebalance_dates:
        rebalance_dates.remove(dates[0])
    return rebalance_dates


def _normalize_tax_config(tax_cfg):
    if tax_cfg is None:
        cfg = TAX_PROFILES['none']
    elif isinstance(tax_cfg, str):
        if tax_cfg not in TAX_PROFILES:
            raise ValueError(f"Unknown tax profile: {tax_cfg}")
        cfg = TAX_PROFILES[tax_cfg]
    else:
        cfg = tax_cfg

    return {
        'stcg_rate': float(cfg.get('stcg_rate', 0.0)),
        'ltcg_rate': float(cfg.get('ltcg_rate', 0.0)),
        'ltcg_days': int(cfg.get('ltcg_days', 365)),
        'stcg_days': int(cfg.get('stcg_days', 0)),
        'ltcg_exemption': float(cfg.get('ltcg_exemption', 0.0)),
        'apply_ltcg_exemption': bool(cfg.get('apply_ltcg_exemption', False))
    }


def _sell_units_fifo(lots, units_to_sell, price, sell_date, tax_cfg):
    remaining = float(units_to_sell)
    proceeds = 0.0
    realized_gain = 0.0
    st_gain = 0.0
    lt_gain = 0.0
    st_loss = 0.0
    lt_loss = 0.0
    new_lots = []

    for lot in lots:
        if remaining <= 1e-12:
            new_lots.append(lot)
            continue

        lot_units = lot['units']
        use_units = min(lot_units, remaining)
        proceeds += use_units * price

        gain = (price - lot['cost_per_unit']) * use_units
        realized_gain += gain
        holding_days = (sell_date - lot['date']).days
        if holding_days >= tax_cfg['ltcg_days']:
            if gain >= 0:
                lt_gain += gain
            else:
                lt_loss += -gain
        elif holding_days >= tax_cfg['stcg_days']:
            if gain >= 0:
                st_gain += gain
            else:
                st_loss += -gain

        remaining -= use_units
        remaining_lot_units = lot_units - use_units
        if remaining_lot_units > 1e-12:
            new_lots.append({
                'units': remaining_lot_units,
                'cost_per_unit': lot['cost_per_unit'],
                'date': lot['date']
            })

    units_sold = units_to_sell - remaining
    return {
        'units_sold': units_sold,
        'gross_proceeds': proceeds,
        'realized_gain': realized_gain,
        'st_gain': st_gain,
        'lt_gain': lt_gain,
        'st_loss': st_loss,
        'lt_loss': lt_loss,
        'lots': new_lots
    }


def _fy_year(date):
    if date.month >= 4:
        return date.year
    return date.year - 1


def _consume_loss_buckets(buckets, amount):
    remaining = float(amount)
    buckets = sorted(buckets, key=lambda x: x['years_left'])
    new_buckets = []

    for bucket in buckets:
        if remaining <= 1e-12:
            new_buckets.append(bucket)
            continue
        use_amt = min(bucket['amount'], remaining)
        bucket['amount'] -= use_amt
        remaining -= use_amt
        if bucket['amount'] > 1e-12:
            new_buckets.append(bucket)

    return remaining, new_buckets


def _roll_forward_buckets(buckets):
    rolled = []
    for bucket in buckets:
        years_left = bucket['years_left']
        if bucket.get('origin') == 'current':
            years_left = 8
        else:
            years_left -= 1
        if years_left > 0 and bucket['amount'] > 1e-12:
            rolled.append({'amount': bucket['amount'], 'years_left': years_left, 'origin': 'carry'})
    return rolled


def _apply_tax_netting(st_gain, st_loss, lt_gain, lt_loss, tax_cfg, carry):
    stcl_buckets = [{'amount': x['amount'], 'years_left': x['years_left'], 'origin': 'carry'} for x in carry['stcl']]
    ltcl_buckets = [{'amount': x['amount'], 'years_left': x['years_left'], 'origin': 'carry'} for x in carry['ltcl']]

    if st_loss > 0:
        stcl_buckets.append({'amount': st_loss, 'years_left': 8, 'origin': 'current'})
    if lt_loss > 0:
        ltcl_buckets.append({'amount': lt_loss, 'years_left': 8, 'origin': 'current'})

    # Offset STCG with STCL (oldest first)
    remaining_stcg, stcl_buckets = _consume_loss_buckets(stcl_buckets, st_gain)
    stcg_taxable = max(remaining_stcg, 0.0)

    # Offset LTCG with LTCL first, then remaining STCL
    remaining_ltcg, ltcl_buckets = _consume_loss_buckets(ltcl_buckets, lt_gain)
    remaining_ltcg, stcl_buckets = _consume_loss_buckets(stcl_buckets, remaining_ltcg)

    taxable_ltcg = max(remaining_ltcg, 0.0)
    if tax_cfg['apply_ltcg_exemption']:
        taxable_ltcg = max(taxable_ltcg - tax_cfg['ltcg_exemption'], 0.0)

    tax_due = taxable_ltcg * tax_cfg['ltcg_rate'] + stcg_taxable * tax_cfg['stcg_rate']

    carry_next = {
        'stcl': _roll_forward_buckets(stcl_buckets),
        'ltcl': _roll_forward_buckets(ltcl_buckets)
    }

    return tax_due, carry_next


def build_rebalanced_portfolio_on_grid(portfolio_def, index_data, common_dates,
                                       rebalance_cfg, tax_cfg=None,
                                       return_details=False):
    """
    Build a portfolio series with threshold-based rebalancing and tax drag.
    Realized gains/losses are accrued during the FY and tax is paid on the first
    trading day of April (India FY).

    Parameters:
    -----------
    portfolio_def : list of (index_name, weight)
    index_data : dict of pd.Series, indexed by Date
    common_dates : iterable of datetime-like
    rebalance_cfg : dict, threshold-based or interval-based configuration.
    tax_cfg : dict or str, keys: stcg_rate, ltcg_rate, stcg_days, ltcg_days,
              ltcg_exemption, apply_ltcg_exemption; or a profile name from TAX_PROFILES
    return_details : bool, include rebalance/tax metadata
    """
    rebalance_cfg = _validate_rebalance_config(rebalance_cfg)
    tax_cfg = _normalize_tax_config(tax_cfg)
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

    cash = 0.0
    last_rebalance_date = None
    portfolio_values = []
    rebalance_log = []
    total_tax_paid = 0.0
    realized = {'st_gain': 0.0, 'st_loss': 0.0, 'lt_gain': 0.0, 'lt_loss': 0.0}
    carry_forward = {'stcl': [], 'ltcl': []}
    pending_tax = 0.0

    last_date = None
    rebalance_dates = set()
    if rebalance_cfg.get('method') == 'interval':
        rebalance_dates = _compute_interval_rebalance_dates(date_index, rebalance_cfg)

    for date in date_index:
        # Pay prior FY tax on the first trading day of new FY (April)
        if last_date is not None and _fy_year(date) != _fy_year(last_date):
            tax_due, carry_forward = _apply_tax_netting(
                realized['st_gain'], realized['st_loss'],
                realized['lt_gain'], realized['lt_loss'],
                tax_cfg, carry_forward
            )
            realized = {'st_gain': 0.0, 'st_loss': 0.0, 'lt_gain': 0.0, 'lt_loss': 0.0}
            pending_tax = tax_due

        if pending_tax > 0:
            note = 'FY tax payment'
            if cash + 1e-12 < pending_tax:
                shortfall = pending_tax - cash
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
                            cash += result['gross_proceeds']
                            realized['st_gain'] += result['st_gain']
                            realized['st_loss'] += result['st_loss']
                            realized['lt_gain'] += result['lt_gain']
                            realized['lt_loss'] += result['lt_loss']

            if cash + 1e-12 < pending_tax:
                pending_tax = cash
                note = 'FY tax payment (partial)'

            cash -= pending_tax
            total_tax_paid += pending_tax
            if return_details:
                rebalance_log.append({
                    'date': date,
                    'tax_paid': pending_tax,
                    'cash_after': cash,
                    'note': note,
                    'event': 'tax_payment'
                })
            pending_tax = 0.0

        prices = price_df.loc[date]
        asset_values = {name: units[name] * prices[name] for name in names}
        total_value = cash + sum(asset_values.values())

        should_rebalance = False
        drift = []
        if date != first_date and total_value > 0:
            if rebalance_cfg.get('method') == 'interval':
                if date in rebalance_dates:
                    should_rebalance = True
            else:
                min_days_between = rebalance_cfg.get('min_days_between', 0)
                if (min_days_between > 0 and last_rebalance_date is not None):
                    if (date - last_rebalance_date).days < min_days_between:
                        should_rebalance = False
                    else:
                        should_rebalance = True
                else:
                    should_rebalance = True

                if should_rebalance:
                    current_weights = {
                        name: asset_values[name] / total_value for name in names
                    }
                    drift = [
                        abs(current_weights[name] - target_weights[i])
                        for i, name in enumerate(names)
                    ]
                    if max(drift) <= rebalance_cfg['drift_threshold']:
                        should_rebalance = False

        if should_rebalance:
            weights_before = {
                name: asset_values[name] / total_value for name in names
            }
            target_values = {
                name: total_value * target_weights[i]
                for i, name in enumerate(names)
            }
            trades = {'sells': {}, 'buys': {}}
            realized_this = {'st_gain': 0.0, 'lt_gain': 0.0, 'st_loss': 0.0, 'lt_loss': 0.0}

            # Sell first to raise cash
            for name in names:
                delta_value = target_values[name] - asset_values[name]
                if delta_value < -1e-12:
                    price = prices[name]
                    units_to_sell = min(units[name], (-delta_value) / price)
                    if units_to_sell > 1e-12:
                        result = _sell_units_fifo(
                            lots[name], units_to_sell, price, date, tax_cfg
                        )
                        units[name] -= result['units_sold']
                        lots[name] = result['lots']
                        cash += result['gross_proceeds']
                        realized['st_gain'] += result['st_gain']
                        realized['st_loss'] += result['st_loss']
                        realized['lt_gain'] += result['lt_gain']
                        realized['lt_loss'] += result['lt_loss']
                        realized_this['st_gain'] += result['st_gain']
                        realized_this['st_loss'] += result['st_loss']
                        realized_this['lt_gain'] += result['lt_gain']
                        realized_this['lt_loss'] += result['lt_loss']
                        trades['sells'][name] = {
                            'units': result['units_sold'],
                            'value': result['gross_proceeds'],
                            'st_gain': result['st_gain'],
                            'lt_gain': result['lt_gain'],
                            'st_loss': result['st_loss'],
                            'lt_loss': result['lt_loss']
                        }

            # Recompute asset values after sells
            asset_values = {name: units[name] * prices[name] for name in names}

            # Buy with available cash (scale if tax drag reduced proceeds)
            desired_buys = {
                name: max(target_values[name] - asset_values[name], 0.0)
                for name in names
            }
            total_buy_value = sum(desired_buys.values())
            if total_buy_value > 1e-12 and cash > 1e-12:
                scale = min(1.0, cash / total_buy_value)
                for name in names:
                    buy_value = desired_buys[name] * scale
                    if buy_value <= 1e-12:
                        continue
                    price = prices[name]
                    units_to_buy = buy_value / price
                    units[name] += units_to_buy
                    cash -= buy_value
                    lots[name].append({
                        'units': units_to_buy,
                        'cost_per_unit': price,
                        'date': date
                    })
                    trades['buys'][name] = {
                        'units': units_to_buy,
                        'value': buy_value
                    }

            last_rebalance_date = date

            if return_details:
                asset_values_after = {name: units[name] * prices[name] for name in names}
                total_value_after = cash + sum(asset_values_after.values())
                weights_after = {
                    name: (asset_values_after[name] / total_value_after if total_value_after > 0 else 0.0)
                    for name in names
                }
                rebalance_log.append({
                    'date': date,
                    'tax_paid': 0.0,
                    'cash_after': cash,
                    'event': 'rebalance',
                    'weights_before': weights_before,
                    'weights_after': weights_after,
                    'trades': trades,
                    'realized': realized_this,
                    'drift_max': max(drift) if drift else 0.0,
                    'rebalance_method': rebalance_cfg.get('method')
                })

        # Store end-of-day value (post-rebalance)
        asset_values = {name: units[name] * prices[name] for name in names}
        total_value = cash + sum(asset_values.values())
        portfolio_values.append({
            'Date': date,
            'Portfolio_Value': total_value
        })
        last_date = date

    result_df = pd.DataFrame(portfolio_values).reset_index(drop=True)
    if return_details:
        return {
            'data': result_df,
            'rebalance_log': rebalance_log,
            'total_tax_paid': total_tax_paid
        }
    return result_df
