import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from datetime import date

def update_transactions(old_transactions:pd.DataFrame,
                        add_transactions:pd.DataFrame,
                        eps = 1e-6):
    new_transactions = old_transactions.sort_values(by='TransactionDate')
    add_transactions = add_transactions.sort_values(by='TransactionDate')
    num_new_rows = len(new_transactions)
    num_add_rows = len(add_transactions)
    add_row_index = 0
    add_row = add_transactions.iloc[add_row_index]
    add_date = add_row['TransactionDate']
    start_new_row_index = new_transactions['TransactionDate'].searchsorted(add_date)
    while add_row_index < num_add_rows:
        add_row = add_transactions.iloc[add_row_index]
        # Iterate starting at new_transactions[new_row_index] until either
        # new_row_index == num_new_rows or
        # new_transactions[new_row_index] == add_transactions.iloc[add_row_index] 
        stop = False
        at_end = False
        new_row_index = start_new_row_index
        while not stop:
            at_end = new_row_index == num_new_rows
            if not at_end:
                if equal_row(new_transactions.iloc[new_row_index], add_row):
                    stop = True
                else:
                    new_row_index = new_row_index + 1
            else:
                stop = True
        if at_end:
            new_transactions = pd.concat([new_transactions, add_row.to_frame().T], ignore_index=True)
        add_row_index = add_row_index + 1
    new_transactions = new_transactions.sort_values(by='TransactionDate',ascending=False)
    return new_transactions

def equal_row(row1, row2, eps = 1e-6):
    result = \
        row1['TransactionDate'] == row2['TransactionDate'] and \
        row1['TransactionType'] == row2['TransactionType'] and \
        row1['SecurityType'] == row2['SecurityType'] and \
        np.abs(row1['Quantity'] - row2['Quantity']) < eps and \
        np.abs(row1['Amount'] - row2['Amount']) < eps and \
        np.abs(row1['Price'] - row2['Price']) < eps
    return result
