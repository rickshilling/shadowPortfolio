import numpy as np

def get_shadow_transactions(all_transactions, shadow):
    # Sign Convention: 
    #   If I buy then transaction_amounts is positive and transaction_quantities is positive
    #   If I sell then transaction_amounts is negative and transaction_quantities is positive
    transactions = dict()
    for shadow_index, shadow_row in shadow.iterrows():
        ticker = shadow_row['Ticker'].replace("*","")
        transaction_amounts = []
        transaction_quantities = []
        transaction_dates = []
        for _, transaction_row in all_transactions.iterrows():
            if ticker == transaction_row['Symbol'] or ticker == transaction_row['TransactionType']:
                transaction_amounts.append(-transaction_row['Amount'])
                transaction_quantities.append(abs(transaction_row['Quantity']))
                transaction_dates.append(transaction_row['TransactionDate'].to_pydatetime().date())
        transactions[ticker] = dict()
        transactions[ticker]['CurrentPrice($)'] = shadow_row['CurrentPrice($)']
        transactions[ticker]['Price-EarningsRatio(X)'] = shadow_row['Price-EarningsRatio(X)']
        transactions[ticker]['Rel PriceStrgth(%)'] = shadow_row['Rel PriceStrgth(%)']
        indices = np.argsort(transaction_dates) # Ensure oldest date first, most recent date last
        transactions[ticker]['transaction_amounts'] = np.take(transaction_amounts, indices).tolist()
        transactions[ticker]['transaction_quantities'] = np.take(transaction_quantities, indices).tolist()
        transactions[ticker]['transaction_dates'] = np.take(transaction_dates, indices).tolist()
        transactions[ticker]['Notes'] = str(shadow_row['Notes']) 
    transactions = refactor_transactions(transactions)
    return transactions

def refactor_transactions(transactions):
    new_transactions = dict()
    new_transactions['CurrentPrice($)'] = dict()
    new_transactions['Price-EarningsRatio(X)'] = dict()
    new_transactions['Notes'] = dict() 
    new_transactions['Rel PriceStrgth(%)'] = dict()
    new_transactions['transaction_amounts'] = dict()
    new_transactions['transaction_quantities'] = dict()
    new_transactions['transaction_dates'] = dict()
    new_transactions['ticker'] = dict()
    for ticker_index, ticker in enumerate(transactions.keys()):
        new_transactions['CurrentPrice($)'][ticker_index] = transactions[ticker]['CurrentPrice($)']
        new_transactions['Price-EarningsRatio(X)'][ticker_index] = transactions[ticker]['Price-EarningsRatio(X)']
        new_transactions['Rel PriceStrgth(%)'][ticker_index] = transactions[ticker]['Rel PriceStrgth(%)']
        new_transactions['transaction_amounts'][ticker_index] = transactions[ticker]['transaction_amounts']
        new_transactions['transaction_quantities'][ticker_index] = transactions[ticker]['transaction_quantities']
        new_transactions['transaction_dates'][ticker_index] = transactions[ticker]['transaction_dates']
        new_transactions['Notes'][ticker_index] = transactions[ticker]['Notes']
        new_transactions['ticker'][ticker_index] = ticker
    new_transactions['num_stocks'] = ticker_index
    return new_transactions

def set_mean_amount_per_day(transactions, reference_date):
    transactions['mean_amount_per_day'] = dict()
    for ticker_index in range(transactions['num_stocks']):
        cumulative_transaction_amounts = np.cumsum(transactions['transaction_amounts'][ticker_index])
        date_differences = np.diff(transactions['transaction_dates'][ticker_index])
        day_differences = [x.days for x in date_differences]
        if transactions['transaction_dates'][ticker_index] != []:
            last_transaction_date = transactions['transaction_dates'][ticker_index][-1]
            last_day_difference = (reference_date - last_transaction_date).days
            day_differences = np.append(day_differences, last_day_difference)
            transaction_amounts_day_product = cumulative_transaction_amounts * day_differences #($)*(day)
            total_transaction_amounts_day_product = np.sum(transaction_amounts_day_product) #($)*(day)
            duration = (reference_date - transactions['transaction_dates'][ticker_index][0]).days #(day)
            mean_amount = total_transaction_amounts_day_product / duration #($)
            transactions['mean_amount_per_day'][ticker_index] = mean_amount/duration
        else:
            transactions['mean_amount_per_day'][ticker_index] = 0
    return transactions

def set_cost_basis_and_sales_and_current_total_amount(transactions):
    for ticker_index in range(transactions['num_stocks']):
        # if transaction_row['Quantity'] > 0:
        #     total_cost_basis = total_cost_basis + transaction_row['Amount']

    #         average_price_per_time = 0
    #     else:
    #         price_paid_per_time = 0
    #         for amount, delta_time in zip(amount, delta_dates):
    #             if delta_time > 0:
    #                 price_paid_per_time = price_paid_per_time + amount/delta_time
    #         average_price_per_time = -price_paid_per_time/len(amount)
        
    #     if delta_dates == []:
    #         average_amount_per_time = 0
    #     else:
    #         time_differences = np.diff(delta_dates)
    #         time_differences = np.insert(time_differences, 0, delta_dates[0])
    #         cumulative_amounts = np.flip(np.cumsum(np.flip(amount)))
    #         time_amount_product = time_differences * cumulative_amounts # (days)*($)
    #         total_time_amount = np.sum(time_amount_product) # (days)*($)
    #         duration = delta_dates[-1] # (days)
    #         average_amount = total_time_amount / duration # ($)
    #         average_amount_per_time = average_amount / duration # ($)/(days)

    #     shadow_transactions[shadow_ticker] = dict()
    #     shadow_transactions[shadow_ticker]['CurrentPrice($)'] = shadow_row['CurrentPrice($)']
    #     shadow_transactions[shadow_ticker]['Price-EarningsRatio(X)'] = shadow_row['Price-EarningsRatio(X)']
    #     shadow_transactions[shadow_ticker]['Rel PriceStrgth(%)'] = shadow_row['Rel PriceStrgth(%)']
    #     shadow_transactions[shadow_ticker]['num_transactions'] = len(delta_dates)
    #     shadow_transactions[shadow_ticker]['amount'] = amount
    #     shadow_transactions[shadow_ticker]['delta_dates'] = delta_dates
    #     shadow_transactions[shadow_ticker]['average_price_per_time'] = average_amount_per_time
    #     shadow_transactions[shadow_ticker]['index'] = entry_index
    #     shadow_transactions[shadow_ticker]['Notes'] = str(shadow_row['Notes'])
    #     shadow_transactions[shadow_ticker]['current_quantity'] = current_quantity
    #     shadow_transactions[shadow_ticker]['quantities'] = quantities
    #     shadow_transactions[shadow_ticker]['total_cost_basis'] = total_cost_basis
    #     shadow_transactions[shadow_ticker]['total_sales'] = total_sales
    #     shadow_transactions[shadow_ticker]['time_differences'] = time_differences
    #     shadow_transactions[shadow_ticker]['cumulative_amounts'] = cumulative_amounts
    #     shadow_transactions[shadow_ticker]['duration'] = duration
    #     shadow_transactions[shadow_ticker]['total_time_amount'] = total_time_amount
    #     entry_index = entry_index + 1
    # shadow_transactions = refactor_shadow_transactions(shadow_transactions)
    # return shadow_transactions

# def refactor_shadow_transactions(shadow_transactions, nmf_number = 1000):
#     max_num_transactions = 0
#     num_stocks = len(shadow_transactions.keys())
#     for ticker in shadow_transactions.keys():
#         max_num_transactions = np.maximum(max_num_transactions,shadow_transactions[ticker]['num_transactions'])
#     current_prices = np.zeros((num_stocks,1))
#     num_transactions = np.zeros((num_stocks,1), dtype=int)
#     amounts = np.zeros((num_stocks,max_num_transactions))
#     delta_times = np.zeros((num_stocks,max_num_transactions))
#     average_price_per_time = np.zeros((num_stocks,1))
#     PEs = np.zeros((num_stocks,1))
#     order = dict()
#     Notes = dict()
#     Rel_PriceStrgth = np.zeros((num_stocks,1))
#     quantities = dict()
#     current_quantities = np.zeros((num_stocks,1))
#     total_cost_basis = np.zeros((num_stocks,1))
#     total_sales = np.zeros((num_stocks,1))
#     duration = np.zeros((num_stocks,1))
#     time_differences = dict()
#     cumulative_amounts = dict()
#     total_time_amount = np.zeros((num_stocks,1)) 
#     time_amount_product = np.zeros((num_stocks,1))
#     good_standing = np.ones((num_stocks,1))
#     for stock_index in range(num_stocks):
#         found = False
#         for ticker in shadow_transactions.keys():
#             if shadow_transactions[ticker]['index'] == stock_index:
#                 key = ticker
#                 found = True
#                 break
#         if found:
#             current_prices[stock_index] = shadow_transactions[key]['CurrentPrice($)']
#             num_transactions[stock_index][0] = shadow_transactions[key]['num_transactions']
#             current_num_transactions = num_transactions[stock_index][0]
#             amounts[stock_index][0:current_num_transactions] = shadow_transactions[key]['amounts'] 
#             delta_times[stock_index][0:current_num_transactions] = shadow_transactions[key]['delta_times'] 
#             average_price_per_time[stock_index] = shadow_transactions[key]['average_price_per_time']
#             if shadow_transactions[key]['Price-EarningsRatio(X)'] == 'nmf':
#                 PEs[stock_index] = nmf_number
#             else:
#                 PEs[stock_index] = shadow_transactions[key]['Price-EarningsRatio(X)']
#             order[stock_index] = key
#             Notes[stock_index] = shadow_transactions[key]['Notes']
#             Rel_PriceStrgth[stock_index] = shadow_transactions[key]['Rel PriceStrgth(%)']
#             current_quantities[stock_index] = shadow_transactions[key]['current_quantity']
#             quantities[stock_index] = shadow_transactions[key]['quantities']
#             total_cost_basis[stock_index] = shadow_transactions[key]['total_cost_basis']
#             total_sales[stock_index] = shadow_transactions[key]['total_sales']
#             time_differences[stock_index] = shadow_transactions[key]['time_differences']
#             cumulative_amounts[stock_index] = shadow_transactions[key]['cumulative_amounts']
#             duration[stock_index] = shadow_transactions[key]['duration']
#             total_time_amount[stock_index] = shadow_transactions[key]['total_time_amount']
#             if ("Earnings probation" in Notes[stock_index]) or ("Exceeds size limit" in Notes[stock_index]):
#                 good_standing[stock_index] = 0
            
#             # time_amount_product[stock_index] = shadow_transactions[key]['time_amount_product']
#         else:
#             error_string = 'Error: ' + str(stock_index) + ' does not have a key in shadow transactions'
#             print(error_string)
#     new_shadow_transactions = dict()        
#     new_shadow_transactions['tickers'] = list(shadow_transactions.keys())
#     new_shadow_transactions['current_prices'] = current_prices
#     new_shadow_transactions['num_transactions'] = num_transactions
#     new_shadow_transactions['amounts'] = amounts
#     new_shadow_transactions['delta_times'] = delta_times
#     new_shadow_transactions['average_price_per_time'] = average_price_per_time
#     new_shadow_transactions['Price-EarningsRatio(X)'] = PEs
#     new_shadow_transactions['order'] = order
#     new_shadow_transactions['num_stocks'] = len(num_transactions)
#     new_shadow_transactions['Notes'] = Notes
#     new_shadow_transactions['Rel PriceStrgth(%)'] = Rel_PriceStrgth
#     new_shadow_transactions['current_quantities'] = current_quantities
#     new_shadow_transactions['current_amounts'] = current_quantities * current_prices
#     new_shadow_transactions['quantities'] = quantities
#     new_shadow_transactions['total_cost_basis'] = total_cost_basis
#     new_shadow_transactions['total_sales'] = total_sales
#     new_shadow_transactions['time_differences'] = time_differences
#     new_shadow_transactions['cumulative_amounts'] = cumulative_amounts
#     new_shadow_transactions['duration'] = duration
#     new_shadow_transactions['total_time_amount'] = total_time_amount
#     new_shadow_transactions['good_standing'] = good_standing
#     # new_shadow_transactions['time_amount_product'] = time_amount_product
#     return new_shadow_transactions
