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

def get_cost_basis_per_time(transaction_history, today, eps = 1e-6):
    headers = transaction_history.columns
    price_headers = headers[1::2]
    dates = []
    for price_header in price_headers:
        day_string = price_header.split()[-1]
        date_format = "%Y-%m-%d"
        datetime_object = datetime.strptime(day_string, date_format).date()
        dates.append(datetime_object)

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
        
        
