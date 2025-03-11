import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import jax
import jax.numpy as jnp
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

def get_shadow_transactions(transactions, shadow, reference_date, eps = 1e-6):
    shadow_transactions = dict()
    for shadow_index, shadow_row in shadow.iterrows():
        shadow_ticker = shadow_row['Ticker'].replace("*","")
        prices = []
        delta_times = []
        for _, transaction_row in transactions.iterrows():
            if shadow_ticker == transaction_row['Symbol']:
                TransactionDate = transaction_row['TransactionDate'].to_pydatetime().date()
                delta_time = (reference_date - TransactionDate).days
                prices.append(transaction_row['Amount'])
                delta_times.append(delta_time)
        if delta_times == []:
            average_price_per_time = 0
        else:
            price_paid_per_time = 0
            for price, delta_time in zip(prices, delta_times):
                if delta_time > 0:
                    price_paid_per_time = price_paid_per_time + price/delta_time
            average_price_per_time = -price_paid_per_time/len(prices)
        shadow_transactions[shadow_ticker] = dict()
        shadow_transactions[shadow_ticker]['current_price'] = shadow_row['CurrentPrice($)']
        shadow_transactions[shadow_ticker]['num_transactions'] = len(delta_times)
        shadow_transactions[shadow_ticker]['prices'] = prices
        shadow_transactions[shadow_ticker]['delta_times'] = delta_times
        shadow_transactions[shadow_ticker]['average_price_per_time'] = average_price_per_time
        shadow_transactions[shadow_ticker]['index'] = shadow_index
    return shadow_transactions

def refactor_shadow_transactions(shadow_transactions):
    max_num_transactions = 0
    num_stocks = len(shadow_transactions.keys())
    for ticker in shadow_transactions.keys():
        max_num_transactions = np.maximum(max_num_transactions,shadow_transactions[ticker]['num_transactions'])
    current_price = np.zeros((num_stocks,1))
    num_transactions = np.zeros((num_stocks,), dtype=int)
    prices = np.zeros((num_stocks,max_num_transactions))
    delta_times = np.zeros((num_stocks,max_num_transactions))
    average_price_per_time = np.zeros((num_stocks,1))
    order = dict()
    for stock_index in range(num_stocks):
        found = False
        for ticker in shadow_transactions.keys():
            if shadow_transactions[ticker]['index'] == stock_index:
                key = ticker
                found = True
                break
        if found:
            current_price[stock_index] = shadow_transactions[key]['current_price']
            num_transactions[stock_index] = shadow_transactions[key]['num_transactions']
            current_num_transactions = num_transactions[stock_index]
            prices[stock_index][0:current_num_transactions] = shadow_transactions[key]['prices'] 
            delta_times[stock_index][0:current_num_transactions] = shadow_transactions[key]['delta_times'] 
            average_price_per_time[stock_index] = shadow_transactions[key]['average_price_per_time']
            order[stock_index] = key
        else:
            error_string = 'Error: ' + str(stock_index) + ' does not have a key in shadow transactions'
            print(error_string)
    shadow_transactions['tickers'] = list(shadow_transactions.keys())
    shadow_transactions['current_price'] = current_price
    shadow_transactions['num_transactions'] = num_transactions
    shadow_transactions['prices'] = prices
    shadow_transactions['delta_times'] = delta_times
    shadow_transactions['average_price_per_time'] = average_price_per_time
    shadow_transactions['order'] = order
    shadow_transactions['num_stocks'] = len(num_transactions)
    return shadow_transactions

def arg_min_variance(shadow_transactions, T=1, limit = 27*7):
    A = jnp.array(shadow_transactions['average_price_per_time'])
    n = jnp.array(shadow_transactions['num_transactions'])
    m = jnp.array(shadow_transactions['num_stocks']) 
    u = jnp.array(shadow_transactions['current_price'])
    # Let k_i be the number of shares for stock i
    # let k = [k_1, k_2, ..., k_m]
    max_k = np.zeros((m,1))
    max_product = 1
    for i in range(m):
        max_k[i] = np.floor(limit/u[i])
        max_product = max_product * max_k[i]
    A_i_ki = jax.jit(jax.tree_util.Partial(A_n_u_T_i_k,A,n,u,T))
    A_k = jax.jit(jax.tree_util.Partial(A_n_u_T_k,A,n,u,T))
    pass

@jax.jit
def A_n_u_T_i_k(A,n,u,T,i,k):
    return (A[i]*n[i] + k[i]*u[i]/T)/(n[i] + 1)

@jax.jit
def A_n_u_T_k(A,n,u,T,k):
    m = A.shape[0]
    result = 0
    for i in range(m):
        result = result + A_n_u_T_i_k(A,n,u,T,i,k)
    result = result/m
    return result