import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from datetime import date

def udpate_transaction_history(transaction_history:pd.DataFrame, 
                               stock_list,
                               new_date,
                               eps = 1e-6):
  _, num_cols = transaction_history.shape
  num_dates = int(num_cols / 2)
  new_transaction_history = transaction_history
  # 3 possiblities:
  #   In stock_list, in transaction_history
  #   In stock_list, out transaction_history
  #  Out stock_list, in  transaction_history
  added_tickers = []
  added_quantities = []
  added_prices = []
  added_transaction_row_indices = []
  for stock_index, stock_ticker_string in enumerate(stock_list['tickers']):
    stock_in_transaction_history = False
    for transaction_row_index, transaction_row in transaction_history.iterrows():
       if stock_ticker_string == transaction_row['Ticker']:
        stock_in_transaction_history = True
        old_quantities = transaction_row[2::2].values.astype(float)
        old_quantity = np.sum(old_quantities)
        new_quantity = stock_list['my_quantities'][stock_index]
        added_quantity_last_time = new_quantity - old_quantity
        added_transaction_row_index = transaction_row_index
        if abs(added_quantity_last_time) > eps:
          new_total_price_paid = stock_list['my_total_prices_paid'][stock_index]
          old_prices = transaction_row[1::2].values.astype(float)        
          old_total_price_paid = jnp.dot(old_quantities, old_prices)
          added_price_paid = (new_total_price_paid - old_total_price_paid)/added_quantity_last_time
        else:
          added_price_paid = 0
        break
    if not stock_in_transaction_history:
      added_quantity_last_time = stock_list['my_quantities'][stock_index]
      added_transaction_row_index = -1
      if abs(added_quantity_last_time) > eps:
        added_price_paid = stock_list['my_total_prices_paid'][stock_index] / added_quantity_last_time
      else:
        added_price_paid = 0
    added_tickers.append(stock_ticker_string)
    added_quantities.append(added_quantity_last_time)
    added_prices.append(added_price_paid)
    added_transaction_row_indices.append(added_transaction_row_index)
  for transaction_row_index, transaction_row in transaction_history.iterrows():
    stock_in_list = False
    for stock_index, stock_ticker_string in enumerate(stock_list['tickers']):
       if stock_ticker_string == transaction_row['Ticker']:
        stock_in_list = True
    if not stock_in_list:
      added_tickers.append(transaction_row['Ticker'])
      added_quantities.append(0)
      added_prices.append(0)
      added_transaction_row_indices.append(transaction_row_index)

  price_string = 'Price paid ' + str(new_date)
  quantity_string = 'Quantity ' + str(new_date)
  new_transaction_history.insert(1,price_string,0)
  new_transaction_history.insert(2,quantity_string,0)
  for (added_ticker, added_quantity, added_price, added_transaction_row) in zip(added_tickers, added_quantities, added_prices, added_transaction_row_indices):
    if added_transaction_row != -1:
      new_transaction_history.iat[added_transaction_row, 1] =  added_price
      new_transaction_history.iat[added_transaction_row, 2] =  added_quantity
    else:
      new_row = {'Ticker': added_ticker, 
                 price_string: added_price,
                 quantity_string: added_quantity}
      new_transaction_history = pd.concat([new_transaction_history, pd.DataFrame([new_row])], ignore_index=True)

  return new_transaction_history
  
  # Rules
  #  1. The variance of the total cost basis of a stock per unit time owned is minimized among stocks with similar P/Es.
  #  2. Stocks with lower P/Es have a higher total cost basis per unit time than stocks with higher P/Es.

def get_average_price_paid_per_time_owned(transaction_history, today, stock_list, eps = 1e-6):
  headers = transaction_history.columns
  price_headers = headers[1::2]
  dates = []
  for price_header in price_headers:
      day_string = price_header.split()[-1]
      datetime_object = datetime.strptime(day_string, "%Y-%m-%d").date()
      dates.append(datetime_object)

  average_price_paid_per_time_owned_list = dict()
  for transaction_row_index, transaction_row in transaction_history.iterrows():
    quantities = transaction_row[2::2].values
    prices = transaction_row[1::2].values
    price_paid_per_time_owned_list = []
    for index, current_date in enumerate(dates):
      day_delta = (today-current_date).days
      if np.abs(day_delta) > 0 and np.abs(quantities[index]) > eps :
        price_paid_per_time_owned = quantities[index]*prices[index]/day_delta
        price_paid_per_time_owned_list.append(price_paid_per_time_owned)
    num_nonzero_transactions = len(price_paid_per_time_owned_list)
    if num_nonzero_transactions > 0:
      average_price_paid_per_time_owned = sum(price_paid_per_time_owned_list)/num_nonzero_transactions
    else:
      average_price_paid_per_time_owned = 0
    average_price_paid_per_time_owned_list[transaction_row['Ticker']] = average_price_paid_per_time_owned
  stock_list["average_price_paid_per_time_owned"] = average_price_paid_per_time_owned_list
  return stock_list

def optimize_by_average_price_per_time_owned(stock_list):
  average_price_paid_per_time_owned_list = stock_list["average_price_paid_per_time_owned"]
  overall_average_price_paid_per_time_owned_list = np.mean(np.array(list(average_price_paid_per_time_owned_list.values())))
  for ticker in average_price_paid_per_time_owned_list.keys():
    average_price_paid_per_time_owned = average_price_paid_per_time_owned_list[ticker]
    pass


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

def reduce_buys(delta_times, buy_prices, total_reduction_amount):
    # Sign Convention:
    #   price > 0 => Refers to a sell
    #   price < 0 => Refers to a buy
    # What happens when total_reduction_amount is more than the total is owned in prices 
    while total_reduction_amount > 0 and len(buy_prices) > 0:
        num_buys = len(buy_prices)
        reduction_amount = total_reduction_amount / num_buys
        total_reduction_amount = 0
        deletion_indices = []
        for buy_index, price in enumerate(buy_prices):
            reduced_price = price + reduction_amount
            if reduced_price > 0:
                deletion_indices.append(buy_index)
                total_reduction_amount = total_reduction_amount + reduced_price
            else:
                buy_prices[buy_index] = reduced_price
        deletion_indices.sort(reverse=True)
        for deletion_index in deletion_indices:
            del buy_prices[deletion_index]
            del delta_times[deletion_index]
    if len(buy_prices) == 0:
        # total_distribution_amount <= 0
        pass
    return delta_times, buy_prices


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
