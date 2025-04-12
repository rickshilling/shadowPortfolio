import numpy as np
import pandas as pd

def get_shadow_transactions(transactions, shadow, reference_date, eps = 1e-6):
    shadow_transactions = dict()
    for shadow_index, shadow_row in shadow.iterrows():
        shadow_ticker = shadow_row['Ticker'].replace("*","")
        amounts = []
        delta_times = []
        quantities = []
        total_cost_basis = 0
        total_sales = 0
        for _, transaction_row in transactions.iterrows():
            if shadow_ticker == transaction_row['Symbol'] or shadow_ticker == transaction_row['TransactionType']:
                TransactionDate = transaction_row['TransactionDate'].to_pydatetime().date()
                delta_time = (reference_date - TransactionDate).days
                amounts.append(transaction_row['Amount'])
                delta_times.append(delta_time)
                quantities.append(transaction_row['Quantity'])
                if transaction_row['Quantity'] > 0:
                    total_cost_basis = total_cost_basis + transaction_row['Amount']
                else:
                    total_sales = total_sales + transaction_row['Amount']
        current_quantity = np.sum(quantities)
        if delta_times == []:
            average_price_per_time = 0
        else:
            price_paid_per_time = 0
            for amount, delta_time in zip(amounts, delta_times):
                if delta_time > 0:
                    price_paid_per_time = price_paid_per_time + amount/delta_time
            average_price_per_time = -price_paid_per_time/len(amounts)
        shadow_transactions[shadow_ticker] = dict()
        shadow_transactions[shadow_ticker]['CurrentPrice($)'] = shadow_row['CurrentPrice($)']
        shadow_transactions[shadow_ticker]['Price-EarningsRatio(X)'] = shadow_row['Price-EarningsRatio(X)']
        shadow_transactions[shadow_ticker]['Rel PriceStrgth(%)'] = shadow_row['Rel PriceStrgth(%)']
        shadow_transactions[shadow_ticker]['num_transactions'] = len(delta_times)
        shadow_transactions[shadow_ticker]['amounts'] = amounts
        shadow_transactions[shadow_ticker]['delta_times'] = delta_times
        shadow_transactions[shadow_ticker]['average_price_per_time'] = average_price_per_time
        shadow_transactions[shadow_ticker]['index'] = shadow_index
        shadow_transactions[shadow_ticker]['Notes'] = str(shadow_row['Notes'])
        shadow_transactions[shadow_ticker]['current_quantity'] = current_quantity
        shadow_transactions[shadow_ticker]['quantities'] = quantities
        shadow_transactions[shadow_ticker]['total_cost_basis'] = total_cost_basis
        shadow_transactions[shadow_ticker]['total_sales'] = total_sales
    shadow_transactions = refactor_shadow_transactions(shadow_transactions)
    return shadow_transactions

def refactor_shadow_transactions(shadow_transactions, nmf_number = 1000):
    max_num_transactions = 0
    num_stocks = len(shadow_transactions.keys())
    for ticker in shadow_transactions.keys():
        max_num_transactions = np.maximum(max_num_transactions,shadow_transactions[ticker]['num_transactions'])
    current_prices = np.zeros((num_stocks,1))
    num_transactions = np.zeros((num_stocks,1), dtype=int)
    amounts = np.zeros((num_stocks,max_num_transactions))
    delta_times = np.zeros((num_stocks,max_num_transactions))
    average_price_per_time = np.zeros((num_stocks,1))
    PEs = np.zeros((num_stocks,1))
    order = dict()
    Notes = dict()
    Rel_PriceStrgth = np.zeros((num_stocks,1))
    quantities = dict()
    current_quantities = np.zeros((num_stocks,1))
    total_cost_basis = np.zeros((num_stocks,1))
    total_sales = np.zeros((num_stocks,1))
    for stock_index in range(num_stocks):
        found = False
        for ticker in shadow_transactions.keys():
            if shadow_transactions[ticker]['index'] == stock_index:
                key = ticker
                found = True
                break
        if found:
            current_prices[stock_index] = shadow_transactions[key]['CurrentPrice($)']
            num_transactions[stock_index][0] = shadow_transactions[key]['num_transactions']
            current_num_transactions = num_transactions[stock_index][0]
            amounts[stock_index][0:current_num_transactions] = shadow_transactions[key]['amounts'] 
            delta_times[stock_index][0:current_num_transactions] = shadow_transactions[key]['delta_times'] 
            average_price_per_time[stock_index] = shadow_transactions[key]['average_price_per_time']
            if shadow_transactions[key]['Price-EarningsRatio(X)'] == 'nmf':
                PEs[stock_index] = nmf_number
            else:
                PEs[stock_index] = shadow_transactions[key]['Price-EarningsRatio(X)']
            order[stock_index] = key
            Notes[stock_index] = shadow_transactions[key]['Notes']
            Rel_PriceStrgth[stock_index] = shadow_transactions[key]['Rel PriceStrgth(%)']
            current_quantities[stock_index] = shadow_transactions[key]['current_quantity']
            quantities[stock_index] = shadow_transactions[key]['quantities']
            total_cost_basis[stock_index] = shadow_transactions[key]['total_cost_basis']
            total_sales[stock_index] = shadow_transactions[key]['total_sales']
        else:
            error_string = 'Error: ' + str(stock_index) + ' does not have a key in shadow transactions'
            print(error_string)
    new_shadow_transactions = dict()        
    new_shadow_transactions['tickers'] = list(shadow_transactions.keys())
    new_shadow_transactions['current_prices'] = current_prices
    new_shadow_transactions['num_transactions'] = num_transactions
    new_shadow_transactions['amounts'] = amounts
    new_shadow_transactions['delta_times'] = delta_times
    new_shadow_transactions['average_price_per_time'] = average_price_per_time
    new_shadow_transactions['Price-EarningsRatio(X)'] = PEs
    new_shadow_transactions['order'] = order
    new_shadow_transactions['num_stocks'] = len(num_transactions)
    new_shadow_transactions['Notes'] = Notes
    new_shadow_transactions['Rel PriceStrgth(%)'] = Rel_PriceStrgth
    new_shadow_transactions['current_quantities'] = current_quantities
    new_shadow_transactions['current_amounts'] = current_quantities * current_prices
    new_shadow_transactions['quantities'] = quantities
    new_shadow_transactions['total_cost_basis'] = total_cost_basis
    new_shadow_transactions['total_sales'] = total_sales
    return new_shadow_transactions
