import numpy as np
import pandas as pd
import pytest

from utils import calculate_cost_basis_per_time, allocate_weekly_investment


def test_weight_is_constant_per_stock_uses_first_transaction_date():
    transactions = pd.DataFrame(
        {
            'Symbol': ['A', 'A', 'B'],
            'Activity Type': ['Bought', 'Bought', 'Bought'],
            'Quantity #': [10, 5, 8],
            'Amount $': [-100.0, -80.0, -50.0],
            'Price $': [10.0, 16.0, 6.25],
            'Activity/Trade Date': [
                pd.Timestamp('2024-01-01'),
                pd.Timestamp('2024-01-10'),
                pd.Timestamp('2024-02-01'),
            ],
        }
    )

    decay_rate = 0.02
    reference_date = pd.Timestamp('2024-01-30')

    result = calculate_cost_basis_per_time(
        transactions,
        decay_rate=decay_rate,
        reference_date=reference_date,
    )

    stock_a = result[result['Ticker'] == 'A'].iloc[0]
    first_days_ago = (reference_date - pd.Timestamp('2024-01-01')).days
    expected_weight = 1 - np.exp(-decay_rate * first_days_ago)
    expected_weighted_cost_per_day = (180.0 * expected_weight) / 9

    assert stock_a['Weighted Cost per Day'] == pytest.approx(expected_weighted_cost_per_day)


def test_sold_activity_counts_as_sales():
    transactions = pd.DataFrame(
        {
            'Symbol': ['A', 'A'],
            'Activity Type': ['Bought', 'Sold'],
            'Quantity #': [10, -5],
            'Amount $': [-100.0, 60.0],
            'Price $': [10.0, 12.0],
            'Activity/Trade Date': [
                pd.Timestamp('2024-01-01'),
                pd.Timestamp('2024-01-15'),
            ],
        }
    )

    result = calculate_cost_basis_per_time(transactions, decay_rate=0, reference_date=pd.Timestamp('2024-01-20'))
    stock_a = result[result['Ticker'] == 'A'].iloc[0]

    assert stock_a['Total Buy Cost'] == 100.0
    assert stock_a['Total Sell Proceeds'] == 60.0
    assert stock_a['Net Cost'] == 40.0


def test_missing_activity_uses_negative_amount_as_buy():
    transactions = pd.DataFrame(
        {
            'Symbol': ['A', 'A'],
            'Activity Type': [None, 'Sell'],
            'Quantity #': [10, -5],
            'Amount $': [-100.0, 60.0],
            'Price $': [10.0, 12.0],
            'Activity/Trade Date': [
                pd.Timestamp('2024-01-01'),
                pd.Timestamp('2024-01-15'),
            ],
        }
    )

    result = calculate_cost_basis_per_time(transactions, decay_rate=0, reference_date=pd.Timestamp('2024-01-20'))
    stock_a = result[result['Ticker'] == 'A'].iloc[0]

    assert stock_a['Total Buy Cost'] == 100.0
    assert stock_a['Total Sell Proceeds'] == 60.0
    assert stock_a['Net Cost'] == 40.0


def test_allocate_weekly_investment_respects_stock_buy_cap():
    cost_basis_df = pd.DataFrame(
        {
            'Ticker': ['A', 'B'],
            'Weighted Cost per Day': [0.1, 0.2],
            'Net Cost': [0.0, 0.0],
            'Total Shares': [0, 0],
            'Days Held': [0, 0],
        }
    )
    price_df = pd.DataFrame(
        {
            'Ticker': ['A', 'B'],
            'CurrentPrice($)': [100.0, 100.0],
        }
    )
    transactions_df = pd.DataFrame(
        columns=['Symbol', 'Activity Type', 'Quantity #', 'Amount $', 'Activity/Trade Date']
    )

    allocation_df, _ = allocate_weekly_investment(
        cost_basis_df,
        price_df,
        weekly_new_amount=500,
        transactions_df=transactions_df,
        decay_rate=0,
        reference_date=pd.Timestamp('2024-01-20'),
        max_stock_buy=250.0,
    )

    merged = allocation_df.merge(price_df, on='Ticker', how='left')
    assert (merged['Shares to Buy'] * merged['CurrentPrice($)']).le(250.0).all()
