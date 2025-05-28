import numpy as np
from utils import get_mean_amount_per_day
from datetime import date

def minimize_variance_of_new_mean_amount_per_day(
        t, 
        end_date, 
        start_date,
        limit = 27*7, 
        eps=1e-6): 
    new_mean_amount_per_day = t['mean_amount_per_day']
    (good_standing_indices, ) = np.where(t['good_standing']>eps)
    new_transaction_quantities = np.zeros((t['num_stocks'],),dtype=int)
    remaining_amount = limit
    while remaining_amount > 0:
        min_good_index = np.argmin(new_mean_amount_per_day[good_standing_indices])
        min_index = good_standing_indices[min_good_index]
        new_transaction_quantities[min_index] = new_transaction_quantities[min_index] + 1
        remaining_amount = remaining_amount - t['CurrentPrice($)'][min_index]
        new_transaction_amount = new_transaction_quantities * t['CurrentPrice($)']
        new_transaction_amounts = t['transaction_amounts']
        for i in range(t['num_stocks']):
            if t['transaction_amounts'][i] != []:
                new_transaction_amounts[i][-1] = t['transaction_amounts'][i][-1] + new_transaction_amount[i]
            else:
                new_transaction_amounts[i] = new_transaction_amount[i]
        new_mean_amount_per_day = get_mean_amount_per_day( new_transaction_amounts, t['transaction_dates'], end_date, start_date=start_date)
    t['new_transaction_amounts'] = new_transaction_amounts
    t['new_mean_amount_per_day'] = new_mean_amount_per_day
    t['new_transaction_quantities'] = new_transaction_quantities
    return t