import numpy as np
from datetime import date

def get_shadow_transactions(all_transactions, shadow):
    # Sign Convention: 
    #   If I buy then transaction_amounts is positive and transaction_quantities is positive
    #   If I sell then transaction_amounts is negative and transaction_quantities is positive
    t = dict() #(t)ransactions
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
        t[ticker] = dict()
        t[ticker]['CurrentPrice($)'] = shadow_row['CurrentPrice($)']
        t[ticker]['Price-EarningsRatio(X)'] = shadow_row['Price-EarningsRatio(X)']
        t[ticker]['Rel PriceStrgth(%)'] = shadow_row['Rel PriceStrgth(%)']
        indices = np.argsort(transaction_dates) # Ensure oldest date first, most recent date last
        t[ticker]['transaction_amounts'] = np.take(transaction_amounts, indices).tolist()
        t[ticker]['transaction_quantities'] = np.take(transaction_quantities, indices).tolist()
        t[ticker]['transaction_dates'] = np.take(transaction_dates, indices).tolist()
        t[ticker]['Notes'] = str(shadow_row['Notes']) 
    t = refactor_transactions(t)
    return t

def refactor_transactions(t): #(t)ransactions
    nt = dict() #(n)ew (t)ransactions
    num_stocks = len(t.keys())
    nt['CurrentPrice($)'] = np.zeros((num_stocks,))
    nt['Price-EarningsRatio(X)'] = np.zeros((num_stocks,))
    nt['Notes'] = dict() 
    nt['Rel PriceStrgth(%)'] = np.zeros((num_stocks,))
    nt['transaction_amounts'] = dict()
    nt['transaction_quantities'] = dict()
    nt['transaction_dates'] = dict()
    nt['ticker'] = dict()
    for i, ticker in enumerate(t.keys()):
        nt['CurrentPrice($)'][i] = t[ticker]['CurrentPrice($)']
        if t[ticker]['Price-EarningsRatio(X)'] != 'nmf':
            nt['Price-EarningsRatio(X)'][i] = t[ticker]['Price-EarningsRatio(X)']
        else:
            nt['Price-EarningsRatio(X)'][i] = 0
        nt['Rel PriceStrgth(%)'][i] = t[ticker]['Rel PriceStrgth(%)']
        nt['transaction_amounts'][i] = t[ticker]['transaction_amounts']
        nt['transaction_quantities'][i] = t[ticker]['transaction_quantities']
        nt['transaction_dates'][i] = t[ticker]['transaction_dates']
        nt['Notes'][i] = t[ticker]['Notes']
        nt['ticker'][i] = ticker
    nt['num_stocks'] = len(t.keys())
    return nt

def get_mean_amount_per_day( \
            transaction_amounts:dict, \
            transaction_dates:dict, \
            reference_date:date,
            eps=1e-6
            ):
    num_stocks = len(transaction_amounts.keys())
    mean_amount_per_day = np.zeros((num_stocks,))
    for i in range(num_stocks):
        cumulative_transaction_amounts = np.cumsum(transaction_amounts[i])
        date_differences = np.diff(transaction_dates[i])
        day_differences = [x.days for x in date_differences]
        if transaction_dates[i] != []:
            last_transaction_date = transaction_dates[i][-1]
            last_day_difference = (reference_date - last_transaction_date).days
            day_differences = np.append(day_differences, last_day_difference)
            transaction_amounts_day_product = cumulative_transaction_amounts * day_differences #($)*(day)
            total_transaction_amounts_day_product = np.sum(transaction_amounts_day_product) #($)*(day)
            duration = (reference_date - transaction_dates[i][0]).days #(day)
            if duration > eps:
                mean_amount = total_transaction_amounts_day_product / duration #($)
                mean_amount_per_day[i] = mean_amount/duration
            else:
                mean_amount_per_day[i] = 0
        else:
            mean_amount_per_day[i] = 0
    return mean_amount_per_day

def set_mean_amount_per_day(t, reference_date): #(t)ransactions
    mean_amount_per_day = get_mean_amount_per_day( t['transaction_amounts'], t['transaction_dates'], reference_date)
    t['mean_amount_per_day'] = mean_amount_per_day
    return t

def set_current_total_value_and_cost_basis_and_sales(t):  #(t)ransactions
    t['current_total_value'] = dict()
    t['cost_basis'] = dict()
    t['sales'] = dict()
    for i in range(t['num_stocks']):
        cost_basis_indices = np.where(np.array(t['transaction_amounts'][i]) > 0)
        sales_indices = np.where(np.array(t['transaction_amounts'][i]) < 0)
        t['cost_basis'][i] = np.sum(np.take(t['transaction_amounts'][i], cost_basis_indices).tolist())
        t['sales'][i] = np.sum(np.take(t['transaction_amounts'][i], sales_indices).tolist())
        cost_basis_quantity = np.sum(np.take(t['transaction_quantities'][i], cost_basis_indices))
        sale_quantity = np.sum(np.take(t['transaction_quantities'][i], sales_indices))
        current_quantity = cost_basis_quantity - sale_quantity
        t['current_total_value'][i] = np.sum(t['CurrentPrice($)'][i]*current_quantity)
    return t

def set_good_standing(t, stocks_to_exclude = []):
    t['good_standing'] = np.ones((t['num_stocks'],))
    for i in range(t['num_stocks']):
        if \
            "Earnings probation" in t['Notes'][i] or \
            "Exceeds size limit" in t['Notes'][i] or \
            t['ticker'][i] in stocks_to_exclude:
            t['good_standing'][i] = 0
    return t