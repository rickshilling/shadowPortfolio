import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from datetime import date
import copy

def update_transactions(old_transactions:pd.DataFrame,
                        add_transactions:pd.DataFrame,
                        eps = 1e-6):
    new_transactions = old_transactions.sort_values(by='TransactionDate')
    add_transactions = add_transactions.sort_values(by='TransactionDate')
    num_new_rows = len(new_transactions)
    num_add_rows = len(add_transactions)
    add_row_index = 0
    add_row = add_transactions.iloc[add_row_index]
    add_date = add_row['TransactionDate']
    start_new_row_index = new_transactions['TransactionDate'].searchsorted(add_date)
    while add_row_index < num_add_rows:
        add_row = add_transactions.iloc[add_row_index]
        # Iterate starting at new_transactions[new_row_index] until either
        # new_row_index == num_new_rows or
        # new_transactions[new_row_index] == add_transactions.iloc[add_row_index] 
        stop = False
        at_end = False
        new_row_index = start_new_row_index
        while not stop:
            at_end = new_row_index == num_new_rows
            if at_end:
                stop = True
            else:
                if equal_row(new_transactions.iloc[new_row_index], add_row):
                    stop = True
                else:
                    new_row_index = new_row_index + 1
        if at_end:
            new_transactions = pd.concat([new_transactions, add_row.to_frame().T], ignore_index=True)
        add_row_index = add_row_index + 1
    new_transactions = new_transactions.sort_values(by='TransactionDate',ascending=False)
    return new_transactions

def equal_row(row1, row2, eps = 1e-6):
    result = \
        row1['TransactionDate'] == row2['TransactionDate'] and \
        row1['TransactionType'] == row2['TransactionType'] and \
        row1['SecurityType'] == row2['SecurityType'] and \
        row1['Symbol'] == row2['Symbol'] and \
        np.abs(row1['Quantity'] - row2['Quantity']) < eps and \
        np.abs(row1['Amount'] - row2['Amount']) < eps and \
        np.abs(row1['Price'] - row2['Price']) < eps
    return result


def set_average_price_paid_per_time_owned(transactions, shadow, today, eps = 1e-6):
    for _, shadow_row in shadow.iterrows():
        shadow_ticker = shadow_row['Ticker'].replace("*","")
        prices = []
        delta_times = []
        for _, transaction_row in transactions.iterrows():
            if shadow_ticker == transaction_row['Symbol']:
                delta_time = (today - transaction_row['TransactionDate'].to_pydatetime().date()).days
                prices.append(transaction_row['Amount'])
                delta_times.append(delta_time)
        if delta_times == []:
            pass
        else:
            delta_times, prices, sum_of_sells = remove_sells(delta_times, prices)
            delta_times, prices = reduce_buys(delta_times, prices, sum_of_sells)

def remove_sells(delta_times, prices, eps = 1e-6):
    # Sign Convention:
    #   price > 0 => Refers to a sell
    #   price < 0 => Refers to a buy
    sorted_indices = np.argsort(delta_times)[::-1]
    delta_times = [delta_times[i] for i in sorted_indices]
    prices = [prices[i] for i in sorted_indices]
    sell_indices = [i for i, price in enumerate(prices) if price > 0]
    sell_indices.sort(reverse=True)
    sum_of_sells = 0
    for sell_index in sell_indices:
        sum_of_sells = sum_of_sells + prices[sell_index] 
        del prices[sell_index]
        del delta_times[sell_index]
    return delta_times, prices, sum_of_sells

def reduce_buys(delta_times, prices, sum_of_sells):
    total_distribution_amount = sum_of_sells
    while total_distribution_amount > 0 and len(prices) > 0:
        num_buys = len(prices)
        distribution_amount = total_distribution_amount / num_buys
        total_distribution_amount = 0
        deletion_indices = []
        for buy_index, price in enumerate(prices):
            reduced_price = price + distribution_amount
            if reduced_price > 0:
                deletion_indices.append(buy_index)
                total_distribution_amount = total_distribution_amount + reduced_price
            else:
                prices[buy_index] = reduced_price
        deletion_indices.sort(reverse=True)
        for deletion_index in deletion_indices:
            del prices[deletion_index]
            del delta_times[deletion_index]
    if len(prices) == 0:
        # total_distribution_amount <= 0
        pass
    return delta_times, prices
