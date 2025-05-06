import numpy as np

def arg_min_variance(t, T=1, limit = 27*7, eps=1e-6): #(t)ransactions
    m = t['num_stocks']
    (good_standing_indices, good_standing_booleans) = np.where(t['good_standing']>eps)
    remaining_amount = limit
    duration = t['duration']
    new_average_price_per_time = (t['average_price_per_time']*duration)/(duration+T)

    time_differences = t['time_differences']
    cumulative_amounts = t['cumulative_amounts']
    new_shares = np.zeros((t['num_stocks'],1),dtype=int)
    while remaining_amount > 0:
        max_good_index = np.argmax(new_average_price_per_time[good_standing_indices])
        max_index = good_standing_indices[max_good_index]
        new_shares[max_index] = new_shares[max_index] + 1
        for i in range(m):
            time_difference = time_differences[i]
            n = len(time_difference)
            new_time_difference = np.insert(time_difference, 0, T)
            last_amount =  - new_shares[i] * t['current_prices'][i]
            new_amount = np.insert(t['amounts'][i][0:n], 0, last_amount)
            new_cumulative_amount = np.flip(np.cumsum(np.flip(new_amount)))
            new_time_amount_product = new_time_difference * new_cumulative_amount # (days)*($)
            new_total_time_amount = np.sum(new_time_amount_product) 
            new_duration = duration[i] + T
            new_average_amount = new_total_time_amount / new_duration # ($)
            new_average_price_per_time[i] = new_average_amount / new_duration # ($)/(days)
            if i == max_index:
                remaining_amount = remaining_amount + last_amount

    t["new_shares"] = new_shares
    t["new_average_price_per_time"] = new_average_price_per_time
    t["new_amount"] = new_shares * t['current_prices']
    return t