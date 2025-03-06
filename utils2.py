import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from datetime import date

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
            delta_times, prices = remove_sells(delta_times, prices)

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
    remaining_sum_of_sells = sum_of_sells
    while remaining_sum_of_sells > 0 and len(prices) > 0:
        remaining_sum_of_sells = remaining_sum_of_sells + prices[0]
        if remaining_sum_of_sells > 0:
            del prices[0]
            del delta_times[0]
        else:
            prices[0] = remaining_sum_of_sells
    return delta_times, prices

# buy_index = se
# return shadow
#   headers = transaction_history.columns
#   price_headers = headers[1::2]
#   dates = []
#   for price_header in price_headers:
#       day_string = price_header.split()[-1]
#       datetime_object = datetime.strptime(day_string, "%Y-%m-%d").date()
#       dates.append(datetime_object)
#   average_price_paid_per_time_owned_list = dict()
#   for transaction_row_index, transaction_row in transaction_history.iterrows():
#     quantities = transaction_row[2::2].values
#     prices = transaction_row[1::2].values
#     price_paid_per_time_owned_list = []
#     for index, current_date in enumerate(dates):
#       day_delta = (today-current_date).days
#       if np.abs(day_delta) > 0 and np.abs(quantities[index]) > eps :
#         price_paid_per_time_owned = quantities[index]*prices[index]/day_delta
#         price_paid_per_time_owned_list.append(price_paid_per_time_owned)
#     num_nonzero_transactions = len(price_paid_per_time_owned_list)
#     if num_nonzero_transactions > 0:
#       average_price_paid_per_time_owned = sum(price_paid_per_time_owned_list)/num_nonzero_transactions
#     else:
#       average_price_paid_per_time_owned = 0
#     average_price_paid_per_time_owned_list[transaction_row['Ticker']] = average_price_paid_per_time_owned
#   stock_list["average_price_paid_per_time_owned"] = average_price_paid_per_time_owned_list