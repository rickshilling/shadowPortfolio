import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from datetime import date

def update_transactions(old_transactions:pd.DataFrame,
                        new_transactions:pd.DataFrame,
                        eps = 1e-6):
    old_transactions = old_transactions.sort_values(by='TransactionDate', ascending=False)
    new_transactions = new_transactions.sort_values(by='TransactionDate', ascending=False)
    new_transactions = old_transactions
    for current_transaction_row_index, current_transaction_row in current_transactions.iterrows():
        found = False
        for old_transaction_row_index, old_transaction_row in old_transactions.iterrows():
            if equal_row(current_transaction_row, old_transaction_row):
                found = True
                break
        if not found:
            new_transactions = pd.concat([new_transactions, current_transaction_row], ignore_index=True)
    pass

def equal_row(row1, row2, eps = 1e-6):
    result = \
        row1['TransactionDate'] == row2['TransactionDate'] and \
        row1['TransactionType'] == row2['TransactionType'] and \
        row1['SecurityType'] == row2['SecurityType'] and \
        np.abs(row1['Quantity'] - row2['Quantity']) < eps and \
        np.abs(row1['Amount'] - row2['Amount']) < eps and \
        np.abs(row1['Price'] - row2['Price']) < eps
    return result