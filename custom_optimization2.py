import numpy as np
import copy
from utils3 import get_mean_amount_per_day

def minimize_variance_of_future_mean_amount_per_day(t, future_date, limit = 27*7, eps=1e-6): #(t)ransactions
    future_mean_amount_per_day = t['mean_amount_per_day']
    (good_standing_indices, _) = np.where(t['good_standing']>eps)
    future_transaction_quantities = np.zeros((t['num_stocks'],1),dtype=int)
    remaining_amount = limit
    while remaining_amount > 0:
        min_good_index = np.argmin(future_mean_amount_per_day[good_standing_indices])
        min_index = good_standing_indices[min_good_index]
        future_transaction_quantities[min_index] = future_transaction_quantities[min_index] + 1
        remaining_amount = remaining_amount - t['CurrentPrice($)'][min_index]
        future_transaction_amount = future_transaction_quantities * t['CurrentPrice($)']
        future_transaction_amounts = copy.deepcopy(t['transaction_amounts'])
        future_transaction_dates = copy.deepcopy(t['transaction_dates'])
        for i in range(t['num_stocks']):
            future_transaction_amounts[i].append(future_transaction_amount[i][0])
            future_transaction_dates[i].append(future_date)
        future_mean_amount_per_day = get_mean_amount_per_day( future_transaction_amounts, future_transaction_dates, future_date)
    t['future_transaction_amounts'] = future_transaction_amounts
    t['future_mean_amount_per_day'] = future_mean_amount_per_day
    t['future_transaction_quantities'] = future_transaction_quantities
    return t