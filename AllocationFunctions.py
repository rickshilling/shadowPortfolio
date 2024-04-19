import numpy as np
import pandas as pd
import math

def make_line(y_first, y_last, num_xs):
  x_first = 0
  x_last = num_xs
  m = (y_last - y_first) / (x_last - x_first)
  b = y_first
  x = np.arange(x_first, x_last)
  y = m * x + b
  return y

def merge_lists(shadow:pd.DataFrame, mine:pd.DataFrame):
  tickers = []
  stock_prices = []
  notes = []
  PEs = []
  my_amounts = []
  my_percentages = []
  my_quantities = []
  for shadow_index, shadow_element in shadow.iterrows():
    shadow_ticker_string = shadow_element['Ticker']
    shadow_ticker_string = shadow_ticker_string.replace("*","")

    tickers.append(shadow_ticker_string)
    stock_prices.append(shadow_element['CurrentPrice($)'])
    note_element = shadow_element['Notes']
    if isinstance(note_element, str):
       notes.append(note_element)
    else:
       notes.append("")
    PE = shadow_element["Price-EarningsRatio(X)"]
    if isinstance(PE, str):
      if "nmf" in PE:
        PEs.append(np.inf)
    else:
      PEs.append(PE)

    my_amount = 0
    my_quantity = 0
    for my_index, my_element in mine.iterrows():
      if my_element['Symbol'] == shadow_ticker_string:
        my_amount = my_element["Value $"]
        my_quantity = my_element["Quantity"]
        break
    my_amounts.append(my_amount)
    my_quantities.append(my_quantity)
  my_percentages = my_amounts / np.sum(my_amounts)
  my_percentages = list(np.round(my_percentages, decimals = 3))

  tickers = np.array(tickers)
  stock_prices = np.array(stock_prices)
  notes = np.array(notes)
  PEs = np.array(PEs)
  my_amounts = np.array(my_amounts)
  my_percentages = np.array(my_percentages)
  my_quantities = np.array(my_quantities)

  num_stocks = np.size(tickers)
  stock_indices = np.arange(num_stocks)

  stock_list = dict()
  stock_list['tickers'] = tickers
  stock_list['stock_prices'] = stock_prices
  stock_list['PEs'] = PEs
  stock_list['notes'] = notes
  stock_list['my_amounts'] = my_amounts
  stock_list['my_percentages'] = my_percentages
  stock_list['num_stocks'] = num_stocks
  stock_list['my_quantities'] = my_quantities
  return stock_list

def get_weighting(stock_list=[],
                  tickers=[],
                  stock_prices=[],
                  PEs=[],
                  notes=[],
                  my_amounts=[],
                  my_percentages=[],
                  max1_to_min1_ratio = 1.5,  
                  min1_to_med2_ratio = 3,
                  med2_to_med3_ratio = 3):
  # Split stocks by: 
  #  1 finite PE and neither "Earnings probation" nor "Exceeds size limit"
  #  2 infinite PE and neither "Earnings probation" nor "Exceeds size limit"
  #  3 "Earnings probation" or "Exceeds size limit"

  # First Sort & rearrange
  indices = np.argsort(stock_list['PEs'])
  tickers = stock_list['tickers'][indices]
  stock_prices = stock_list['stock_prices'][indices]
  PEs = stock_list['PEs'][indices]
  notes = stock_list['notes'][indices]
  my_amounts = stock_list['my_amounts'][indices]
  my_percentages = stock_list['my_percentages'][indices]
  my_quantities = stock_list['my_quantities'][indices]

  indices1 = []
  indices2 = []
  indices3 = []
  for index, (PE, note) in enumerate(zip(PEs,notes)):
      if ("Earnings probation" in note) or ("Exceeds size limit" in note):
        indices3.append(index)
      elif PE == math.inf:
        indices2.append(index)
      elif PE < math.inf:
        indices1.append(index)
      else:
        pass
  indices = np.concatenate( (indices1,indices2,indices3) )
  weights1 = make_line(y_first=max1_to_min1_ratio, 
                                          y_last=1, 
                                          num_xs = len(indices1))
  weights2 = make_line(y_first=1/min1_to_med2_ratio, 
                                          y_last=1/min1_to_med2_ratio, 
                                          num_xs = len(indices2))
  weights3 = make_line(y_first=1/min1_to_med2_ratio/med2_to_med3_ratio, 
                                          y_last=1/min1_to_med2_ratio/med2_to_med3_ratio, 
                                          num_xs = len(indices3))
  weights = np.concatenate( (weights1,weights2,weights3) )
  target_percentages = weights / np.sum(weights)

  tickers = tickers[indices]
  stock_prices = stock_prices[indices]
  PEs = PEs[indices]
  notes = notes[indices]
  my_amounts = my_amounts[indices]
  my_percentages = my_percentages[indices]
  my_quantities = my_quantities[indices]

  stock_list['tickers'] = tickers
  stock_list['stock_prices'] = stock_prices
  stock_list['PEs'] = PEs
  stock_list['notes'] = notes
  stock_list['my_amounts'] = my_amounts
  stock_list['my_percentages'] = my_percentages
  stock_list['target_percentages'] = target_percentages
  stock_list['my_quantities'] = my_quantities
  return stock_list

def distribute(stock_list, contribution):
  my_amount = stock_list['my_amounts'].sum()
  target_amount = my_amount + contribution
  stock_list['target_amounts'] = target_amount * stock_list['target_percentages']
  stock_list['difference_amounts'] = stock_list['target_amounts'] - stock_list['my_amounts']
  pass

def polarize_by_difference(stock_list):
  stock_list['positive'] = dict()
  stock_list['negative'] = dict()
  for polarity in ['positive','negative']:
    stock_list[polarity]['tickers'] = []
    stock_list[polarity]['stock_prices'] = []
    stock_list[polarity]['PEs'] = []
    stock_list[polarity]['notes'] = []
    stock_list[polarity]['my_amounts'] = []
    stock_list[polarity]['my_percentages'] = []
    stock_list[polarity]['target_percentages'] = []
    stock_list[polarity]['num_stocks'] = 0
    stock_list[polarity]['percentage_differences'] = []
  stock_list['percentage_differences'] = stock_list['target_percentages'] - stock_list['my_percentages']
  for i, percent_difference in enumerate(stock_list['percentage_differences']):
    if percent_difference >= 0:
      polarity = 'positive'
    else:
      polarity = 'negative'
    stock_list[polarity]['tickers'].append(stock_list['tickers'][i])
    stock_list[polarity]['stock_prices'].append(stock_list['stock_prices'][i])
    stock_list[polarity]['PEs'].append(stock_list['PEs'][i])
    stock_list[polarity]['notes'].append(stock_list['notes'][i])
    stock_list[polarity]['my_amounts'].append(stock_list['my_amounts'][i])
    stock_list[polarity]['my_percentages'].append(stock_list['my_percentages'][i])
    stock_list[polarity]['target_percentages'].append(stock_list['target_percentages'][i])
    stock_list[polarity]['num_stocks'] += 1
    stock_list[polarity]['percentage_differences'].append(percent_difference)
  stock_list['positive']['normalized_percentage_differences'] = stock_list['positive']['percentage_differences'] / np.sum(stock_list['positive']['percentage_differences'])
  stock_list['negative']['normalized_percentage_differences'] = stock_list['negative']['percentage_differences'] / np.sum(stock_list['negative']['percentage_differences'])
  return stock_list
  
def get_optimal_allocation(stock_list, positive_delta_contribution, negative_delta_contribution):
    for polarity in ['negative','positive']:
        if polarity == 'positive':
            delta_contribution = positive_delta_contribution
        else:
            delta_contribution = negative_delta_contribution
            # delta_contribution = -negative_delta_contribution
        target_delta_contributions = delta_contribution * stock_list[polarity]['normalized_percentage_differences']
        stock_list[polarity]['target_delta_contributions'] = target_delta_contributions
        stock_list[polarity]['target_amounts']=stock_list[polarity]['my_amounts'] + stock_list[polarity]['target_delta_contributions']
        
        low_quantities = np.floor(target_delta_contributions / stock_list[polarity]['stock_prices'])
        target_amount = target_delta_contributions.sum()
        current_best_deltas = np.zeros(stock_list[polarity]['num_stocks'])
        current_best_amount_allocation = np.multiply(low_quantities+current_best_deltas,stock_list[polarity]['stock_prices'])
        current_best_amount = np.sum(current_best_amount_allocation)
        num_states = 2**stock_list[polarity]['num_stocks']
        for state in range(num_states):
            current_number = state
            current_deltas = np.zeros(stock_list[polarity]['num_stocks'])
            current_objective = 0
            for i in range(stock_list[polarity]['num_stocks']):
                delta_index = stock_list[polarity]['num_stocks'] - 1 - i
                current_delta = current_number % 2
                current_number = current_number >> 1
                current_deltas[delta_index] = current_delta
                current_low_quantity = low_quantities[delta_index]
                current_unit_price = stock_list[polarity]['stock_prices'][delta_index]
                current_single_amount = (current_low_quantity + current_delta)*current_unit_price
                current_objective += (target_delta_contributions[delta_index] - current_single_amount)**2
            current_amount_allocation = np.multiply(low_quantities + current_deltas, stock_list[polarity]['stock_prices'])
            current_amount = np.sum(current_amount_allocation)
            if state == 0:
                current_best_objective = current_objective
            if current_best_objective < current_objective and current_amount <= target_amount:
                current_best_objective = current_objective
                current_best_deltas = current_deltas
                current_best_amount = current_amount
                current_best_amount_allocation = current_amount_allocation
        best_allocation = current_best_amount_allocation
        stock_list[polarity]['optimized_quantities'] = low_quantities + current_best_deltas
        stock_list[polarity]['optimized_target_amounts'] = current_best_amount_allocation
    return stock_list

def optimize_by_difference(stock_list, postive_contribution, negative_contribution):
    my_amounts = stock_list['my_amounts']
    my_amount = my_amounts.sum()
    target_amount = my_amount + postive_contribution 
    target_amounts = target_amount * stock_list['target_percentages']
    difference_amounts = target_amounts - my_amounts
    indices_to_increase = np.nonzero(difference_amounts >= 0)
    indices_to_decrease = np.nonzero(difference_amounts < 0)
    my_amounts_to_increase = my_amounts[indices_to_increase]
    my_amounts_to_decrease = my_amounts[indices_to_decrease]
    my_amount_to_increase = my_amounts_to_increase.sum()
    my_amount_to_decrease = my_amounts_to_decrease.sum()
    target_amount_to_increase = my_amount_to_increase + postive_contribution
    target_amount_to_decrease = my_amount_to_decrease - negative_contribution
    target_percentages_to_increase = stock_list['target_percentages'][indices_to_increase] / stock_list['target_percentages'][indices_to_increase].sum()
    target_percentages_to_decrease = stock_list['target_percentages'][indices_to_decrease] / stock_list['target_percentages'][indices_to_decrease].sum() 
    target_amounts_to_increase = my_amounts_to_increase + postive_contribution * target_percentages_to_increase
    target_amounts_to_decrease = my_amounts_to_decrease - negative_contribution * target_percentages_to_decrease
    stock_list['optimized_amounts'] = np.zeros(stock_list['num_stocks'])
    stock_list['optimized_quantities'] = np.zeros(stock_list['num_stocks'])
    stock_list['optimized_delta_quantities'] = np.zeros(stock_list['num_stocks'])
    stock_list['target_amounts'] = np.zeros(stock_list['num_stocks'])
    stock_list['target_amounts'][indices_to_increase] = target_amounts_to_increase
    stock_list['target_amounts'][indices_to_decrease] = target_amounts_to_decrease

    num_stocks_to_increase = np.size(indices_to_increase)
    num_states_to_increase = 2**num_stocks_to_increase
    stock_prices_to_increase = stock_list['stock_prices'][indices_to_increase]
    base_quantities_to_increase = np.int32(target_amounts_to_increase / stock_prices_to_increase)
    states_to_increase = np.arange(num_states_to_increase)
    delta_quantities_to_increase = np.zeros((num_states_to_increase,num_stocks_to_increase),dtype=np.int8)
    for i in range(num_stocks_to_increase):
        stock_index_to_increase = num_stocks_to_increase - 1 - i
        delta_quantities_to_increase[:,stock_index_to_increase] = states_to_increase % 2
        states_to_increase = states_to_increase >> 1
    quantities_to_increase = base_quantities_to_increase + delta_quantities_to_increase
    amounts_to_increase = np.multiply(quantities_to_increase, stock_prices_to_increase)
    amount_to_increase = amounts_to_increase.sum(axis = 1)
    differences = target_amounts_to_increase - amounts_to_increase
    scores = np.multiply(differences,differences).sum(axis=1)
    admissible_indices = np.where(amount_to_increase < target_amount_to_increase)
    admissible_max_index = np.argmax(scores[admissible_indices])
    max_index = admissible_indices[0][admissible_max_index]
    stock_list['optimized_amounts'][indices_to_increase] = amounts_to_increase[max_index,:]
    stock_list['optimized_quantities'][indices_to_increase] = quantities_to_increase[max_index,:]
    stock_list['optimized_delta_quantities'][indices_to_increase] = stock_list['optimized_quantities'][indices_to_increase]- stock_list['my_quantities'][indices_to_increase]

    num_stocks_to_decrease = np.size(indices_to_decrease)
    num_states_to_decrease = 2**num_stocks_to_decrease
    stock_prices_to_decrease = stock_list['stock_prices'][indices_to_decrease]
    base_quantities_to_decrease = np.ceil(target_amounts_to_decrease / stock_prices_to_decrease)
    states_to_decrease = np.arange(num_states_to_decrease)
    delta_quantities_to_decrease = np.zeros((num_states_to_decrease,num_stocks_to_decrease),dtype=np.int8)
    for i in range(num_stocks_to_decrease):
        stock_index_to_decrease = num_stocks_to_decrease - 1 - i
        delta_quantities_to_decrease[:,stock_index_to_decrease] = states_to_decrease % 2
        states_to_decrease = states_to_decrease >> 1
    quantities_to_decrease = base_quantities_to_decrease - delta_quantities_to_decrease
    amounts_to_decrease = np.multiply(quantities_to_decrease, stock_prices_to_decrease)
    amount_to_decrease = amounts_to_decrease.sum(axis = 1)
    differences = target_amounts_to_decrease - amounts_to_decrease
    scores = np.multiply(differences,differences).sum(axis=1)
    admissible_indices = np.where(amount_to_decrease > target_amount_to_decrease)
    admissible_max_index = np.argmax(scores[admissible_indices])
    max_index = admissible_indices[0][admissible_max_index]
    stock_list['optimized_amounts'][indices_to_decrease] = amounts_to_decrease[max_index,:]
    stock_list['optimized_quantities'][indices_to_decrease] = quantities_to_decrease[max_index,:]
    stock_list['optimized_delta_quantities'][indices_to_decrease] = stock_list['optimized_quantities'][indices_to_decrease]- stock_list['my_quantities'][indices_to_decrease]
    return stock_list