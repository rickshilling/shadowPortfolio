import numpy as np
from datetime import date, datetime
from typing import List

def get_shadow_transactions(all_transactions, shadow, stocks_to_exclude=[]):
    # Sign Convention: 
    #   If I buy then transaction_amounts is positive and transaction_quantities is positive
    #   If I sell then transaction_amounts is negative and transaction_quantities is positive
    t = dict() #(t)ransactions
    for shadow_index, shadow_row in shadow.iterrows():
        ticker = shadow_row['Ticker'].replace("*","")
        if ticker in stocks_to_exclude:
            continue
        transaction_amounts = []
        transaction_quantities = []
        transaction_dates = []
        for _, transaction_row in all_transactions.iterrows():
            if ticker == transaction_row['Symbol'] or ticker == transaction_row['Activity Type']:
                transaction_amount = transaction_row['Quantity #'] * transaction_row['Price $']
                # transaction_amounts.append(-transaction_row['Amount'])
                if abs(transaction_amount) > 1e-3: 
                    transaction_amounts.append(transaction_amount)
                    transaction_quantities.append(abs(transaction_row['Quantity #']))
                    transaction_date_var = transaction_row['Transaction Date']
                    if isinstance(transaction_date_var, str):
                        dt_object = datetime.strptime(transaction_date_var, "%m/%d/%y")
                        final_date = dt_object.date()
                    else:
                        final_date = transaction_date_var.date()#.to_pydatetime()
                    transaction_dates.append(final_date)
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

def get_amount_per_day( \
            transaction_amounts:dict, \
            transaction_dates:dict, \
            end_date:date,
            start_dates, \
            eps=1e-6
            ):
    num_stocks = len(transaction_amounts.keys())
    amount_per_day = np.zeros((num_stocks,))
    # assert(start_date <= end_date)
    for i in range(num_stocks):
        duration = (end_date - start_dates[i]).days
        assert(np.array_equal(np.sort(transaction_dates[i]),transaction_dates[i]))
        num_transactions = len(transaction_dates[i])
        if transaction_dates[i] == []:
            amount_per_day[i] = 0
            continue

        # Find first transaction on or after the start date
        first_transaction_index_on_or_after_start_date = 0
        stop = False
        while not stop:
            if first_transaction_index_on_or_after_start_date == num_transactions:
                stop = True
            else:
                if (transaction_dates[i][first_transaction_index_on_or_after_start_date] < start_dates[i]):
                    first_transaction_index_on_or_after_start_date = first_transaction_index_on_or_after_start_date + 1
                else:
                    stop = True

        # Find last transaction before or on end date
        index = first_transaction_index_on_or_after_start_date
        stop = False
        while not stop:
            if index == num_transactions:
                stop = True
            else:
                if (transaction_dates[i][index] > end_date):
                    stop = True
                else:
                    index = index + 1
        last_transaction_index_before_or_on_end_date = max(0,index - 1)

        start_index = max(0,min(num_transactions-1,first_transaction_index_on_or_after_start_date))
        end_index = max(0,min(num_transactions-1,last_transaction_index_before_or_on_end_date))

        transaction_sum = np.sum(transaction_amounts[i][start_index:(end_index+1)])
        amount_per_day[i] = transaction_sum/duration
    return amount_per_day

def get_weighted_amount_per_day( \
            transaction_amounts:dict, \
            transaction_dates:dict, \
            end_date:date,
            start_dates, \
            tau = 1e0,\
            eps=1e-6,
            ):
    # We want the amount bought per day to be the same between any two stocks. 
    # This means, 
    #   if stocks x,y owned for time T(x)>T(y) have amounts A(x)>A(y)
    #   then the amounts per day A(x)/T(x)=A(y)/T(y).    
    # However, we also prefer to have newly owned stocks to have a catch-up period.  
    # So we weigh down longer owned stocks versus shorted owned stocks.  
    # This means, we need to introduce W()>0 such that 
    #     W(x)A(x)/T(x)<W(y)A(y)/T(y) => W(x)<W(y).
    # We also want some additional properties.  
    # As time increases we want to diminish the weight.  
    # This means for stocks x,y,z, T(x)>T(y)>T(z)
    #   if (T(x)-T(y)) = (T(y)-T(z)) then W(x) < W(y) < W(z).
    # Start with W(.) to be a function of time:
    #   W(t)=1-exp(-t/tau) => W(x) = 1-exp(-T(x)/tau)

    num_stocks = len(transaction_amounts.keys())
    amount_per_day = np.zeros((num_stocks,))
    # assert(start_date <= end_date)
    for i in range(num_stocks):
        duration = (end_date - start_dates[i]).days
        assert(np.array_equal(np.sort(transaction_dates[i]),transaction_dates[i]))
        num_transactions = len(transaction_dates[i])
        if transaction_dates[i] == []:
            amount_per_day[i] = 0
            continue

        # Find first transaction on or after the start date
        first_transaction_index_on_or_after_start_date = 0
        stop = False
        while not stop:
            if first_transaction_index_on_or_after_start_date == num_transactions:
                stop = True
            else:
                if (transaction_dates[i][first_transaction_index_on_or_after_start_date] < start_dates[i]):
                    first_transaction_index_on_or_after_start_date = first_transaction_index_on_or_after_start_date + 1
                else:
                    stop = True

        # Find last transaction before or on end date
        index = first_transaction_index_on_or_after_start_date
        stop = False
        while not stop:
            if index == num_transactions:
                stop = True
            else:
                if (transaction_dates[i][index] > end_date):
                    stop = True
                else:
                    index = index + 1
        last_transaction_index_before_or_on_end_date = max(0,index - 1)

        start_index = max(0,min(num_transactions-1,first_transaction_index_on_or_after_start_date))
        end_index = max(0,min(num_transactions-1,last_transaction_index_before_or_on_end_date))

        transaction_sum = np.sum(transaction_amounts[i][start_index:(end_index+1)])
        weight = 1-np.exp(-duration/365/tau)
        amount_per_day[i] = weight*transaction_sum/duration
    return amount_per_day

def set_amount_per_day(t, end_date, start_dates): #(t)ransactions
    amount_per_day = get_amount_per_day( t['transaction_amounts'], t['transaction_dates'], end_date, start_dates)
    t['amount_per_day'] = amount_per_day
    return t

def set_weighted_amount_per_day(t, end_date, start_dates, tau=1e0): #(t)ransactions
    amount_per_day = get_weighted_amount_per_day( t['transaction_amounts'], t['transaction_dates'], end_date, start_dates, tau)
    t['amount_per_day'] = amount_per_day
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

def set_date_of_first_purchase(t):
    t['date_of_first_purchase'] = {}
    for i in range(t['num_stocks']):
        # if t['transaction_dates'][i] != []:
        if t['good_standing'][i]:
            t['date_of_first_purchase'][i] = t['transaction_dates'][i][0]
        # else:
        #     t['date_of_first_purchase'][i] = []
    return t

def get_tau(weight, duration_in_days):
    # Solves for:
    #  weight = 1 - exp(-duration_in_days/tau)
    tau = -duration_in_days / np.log(1-weight)
    return tau