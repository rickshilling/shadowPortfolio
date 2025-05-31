import numpy as np
from utils import get_amount_per_day
from datetime import date
import copy

def minimize_variance_of_new_amount_per_day(
        t, #(t)ransactions 
        end_date, 
        new_transaction_date,
        start_date,
        limit = 27*7, 
        eps=1e-6): 
    new_amount_per_day = t['amount_per_day']
    (good_standing_indices, ) = np.where(t['good_standing']>eps)
    new_transaction_quantities = np.zeros((t['num_stocks'],),dtype=int)
    new_transaction_dates = t['transaction_dates']
    for i in range(t['num_stocks']):
        new_transaction_dates[i].append(new_transaction_date)
    remaining_amount = limit
    while remaining_amount > 0:
        min_good_index = np.argmin(new_amount_per_day[good_standing_indices])
        min_index = good_standing_indices[min_good_index]
        new_transaction_quantities[min_index] = new_transaction_quantities[min_index] + 1
        remaining_amount = remaining_amount - t['CurrentPrice($)'][min_index]
        new_transaction_amount = new_transaction_quantities * t['CurrentPrice($)']
        new_transaction_amounts = copy.deepcopy(t['transaction_amounts'])
        for i in range(t['num_stocks']):
            if i == 6:
                pass
            new_transaction_amounts[i].append(new_transaction_amount[i])
        new_amount_per_day = get_amount_per_day( new_transaction_amounts, new_transaction_dates, end_date, start_date=start_date)
    t['new_transaction_amounts'] = new_transaction_amounts
    t['new_amount_per_day'] = new_amount_per_day
    t['new_transaction_quantities'] = new_transaction_quantities
    return t