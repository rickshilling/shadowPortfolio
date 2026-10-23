import pandas as pd
import numpy as np
from scipy.optimize import fsolve, least_squares


def _is_buy_transaction(activity, amount):
    """Return True when a transaction is a purchase.

    If the activity label is missing or ambiguous, a negative cash amount is treated
    as a buy. Positive amounts are treated as sells unless the label explicitly says
    otherwise.
    """
    activity_text = str(activity).upper() if pd.notna(activity) else ""

    if 'BOUGHT' in activity_text or 'BUY' in activity_text or 'OPEN' in activity_text:
        return True
    if 'SELL' in activity_text or 'SOLD' in activity_text or 'CLOSE' in activity_text:
        return False

    if pd.isna(amount):
        return False

    amount_value = pd.to_numeric(amount, errors='coerce')
    if pd.isna(amount_value):
        return False

    return amount_value < 0


def _is_sell_transaction(activity, amount):
    """Return True when a transaction is a sale."""
    activity_text = str(activity).upper() if pd.notna(activity) else ""

    if 'SELL' in activity_text or 'SOLD' in activity_text or 'CLOSE' in activity_text:
        return True
    if 'BOUGHT' in activity_text or 'BUY' in activity_text or 'OPEN' in activity_text:
        return False

    if pd.isna(amount):
        return False

    amount_value = pd.to_numeric(amount, errors='coerce')
    if pd.isna(amount_value):
        return False

    return amount_value > 0


def calculate_cost_basis_per_time(transactions, decay_rate=0.01, reference_date=None):
    """
    Calculate cost basis per unit time for each stock.

    Args:
        transactions: DataFrame with transaction data including:
            - Symbol: Stock ticker
            - Activity Type: Type of transaction (e.g., 'Bought', 'Sell')
            - Quantity #: Number of shares
            - Amount $: Transaction amount
            - Activity/Trade Date: Date of transaction
        decay_rate: Rate for inverted exponential decay (default 0.01). Higher values give
                   more weight to older transactions.
        reference_date: Date to use as reference for decay calculations (default: current date)

    Returns:
        DataFrame with columns:
            - Ticker: Stock symbol
            - Total Buy Cost: Total cost of all buy transactions
            - Total Sell Proceeds: Total proceeds from all sell transactions
            - Net Cost: Total buy cost - total sell proceeds
            - Total Shares: Total shares bought
            - Days Held: Number of days from first to last transaction
            - Cost per Day: Net cost / days held
            - Weighted Cost per Day: Net cost with exponential decay weighting / days held
    """
    results = []

    # Use reference_date if provided, otherwise use current date
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()
    else:
        reference_date = pd.Timestamp(reference_date).normalize()

    for ticker in transactions['Symbol'].unique():
        stock_txns = transactions[transactions['Symbol'] == ticker].sort_values('Activity/Trade Date')

        total_buy_cost = 0
        total_sell_proceeds = 0
        total_shares = 0
        total_days = 0
        weighted_net_cost = 0

        if len(stock_txns) > 0:
            first_transaction_date = stock_txns['Activity/Trade Date'].min()
            days_ago_from_first_transaction = (reference_date - first_transaction_date).days

            if np.isclose(decay_rate, 0):
                stock_weight = 1.0
            else:
                stock_weight = 1 - np.exp(-decay_rate * days_ago_from_first_transaction)

        for _, txn in stock_txns.iterrows():
            activity = txn.get('Activity Type')
            quantity = txn['Quantity #']
            price = txn['Price $']
            amount = txn['Amount $']

            # Handle NaN values
            if pd.isna(quantity) or pd.isna(amount):
                continue

            if _is_buy_transaction(activity, amount):
                # Buying shares - add to cost basis with weighting
                total_buy_cost += abs(amount)
                weighted_net_cost += abs(amount) * stock_weight
                total_shares += quantity
            elif _is_sell_transaction(activity, amount):
                # Selling shares - track proceeds with weighting
                total_sell_proceeds += abs(amount)
                weighted_net_cost -= abs(amount) * stock_weight

        # Calculate net cost (buys - sells)
        net_cost = total_buy_cost - total_sell_proceeds

        # Calculate total time period (from first buy to last transaction or current date)
        if len(stock_txns) > 0:
            first_date = stock_txns['Activity/Trade Date'].min()
            last_date = stock_txns['Activity/Trade Date'].max()
            total_days = (last_date - first_date).days

            if total_days > 0 and total_shares > 0:
                cost_per_day = net_cost / total_days
                weighted_cost_per_day = weighted_net_cost / total_days

                results.append({
                    'Ticker': ticker,
                    'Total Buy Cost': total_buy_cost,
                    'Total Sell Proceeds': total_sell_proceeds,
                    'Net Cost': net_cost,
                    'Total Shares': total_shares,
                    'Days Held': total_days,
                    'Cost per Day': cost_per_day,
                    'Weighted Cost per Day': weighted_cost_per_day
                })

    return pd.DataFrame(results)


def allocate_weekly_investment(cost_basis_df, price_df, weekly_new_amount, transactions_df, decay_rate=0.01, reference_date=None, max_stock_buy=250.0):
    """
    Allocate weekly investment by buying shares of stocks with lowest weighted cost per day.
    Updates weighted cost after each purchase to reflect new shares by recalculating from original transactions.

    Args:
        cost_basis_df: DataFrame from calculate_cost_basis_per_time with columns:
            - Ticker: Stock symbol
            - Weighted Cost per Day: Weighted cost metric
            - Net Cost: Net cost of position
            - Total Shares: Total shares owned
            - Days Held: Number of days held
        price_df: DataFrame with current prices including:
            - Ticker: Stock symbol
            - CurrentPrice($): Current stock price
        weekly_new_amount: Total amount in dollars to invest
        transactions_df: Original transaction DataFrame with columns:
            - Symbol: Stock ticker
            - Activity Type: Type of transaction
            - Quantity #: Number of shares
            - Amount $: Transaction amount
            - Activity/Trade Date: Date of transaction
        decay_rate: Decay rate for weighting (default 0.01)
        reference_date: Date to use as reference for decay calculations (default: current date)
        max_stock_buy: Maximum dollar amount to allocate to any single stock in this
            weekly cycle. Once a stock reaches this cap, it is skipped for the rest of
            the cycle. Defaults to 250. Use None to disable the cap.

    Returns:
        tuple: (allocation_df, updated_cost_basis_df)
            - allocation_df: DataFrame with columns:
                - Ticker: Stock symbol
                - Shares to Buy: Number of shares to buy
            - updated_cost_basis_df: DataFrame with updated weighted costs after all purchases
    """
    # Use reference_date if provided, otherwise use current date
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()
    else:
        reference_date = pd.Timestamp(reference_date).normalize()

    # Create a mapping of ticker to current price
    price_map = dict(zip(price_df['Ticker'], price_df['CurrentPrice($)']))

    # Create a dictionary to track shares to buy for each stock
    shares_to_buy = {ticker: 0 for ticker in cost_basis_df['Ticker']}
    allocated_per_stock = {ticker: 0.0 for ticker in cost_basis_df['Ticker']}

    # Make a copy of cost_basis_df that we can modify
    cost_basis_df = cost_basis_df.copy()

    # Store original transactions for each ticker
    ticker_transactions = {}
    for ticker in cost_basis_df['Ticker']:
        ticker_transactions[ticker] = transactions_df[transactions_df['Symbol'] == ticker].copy()

    remaining_amount = weekly_new_amount

    while remaining_amount > 0:
        # Filter to stocks we can afford and that are below the per-stock cap
        affordable_df = cost_basis_df.copy()
        affordable_df['Current Price'] = affordable_df['Ticker'].map(price_map)
        affordable_df = affordable_df[
            (affordable_df['Current Price'].notna()) &
            (affordable_df['Current Price'] <= remaining_amount)
        ]

        if max_stock_buy is not None:
            affordable_df = affordable_df[
                affordable_df.apply(
                    lambda row: allocated_per_stock.get(row['Ticker'], 0.0) + row['Current Price'] <= max_stock_buy,
                    axis=1,
                )
            ]

        if len(affordable_df) == 0:
            break

        # Find the stock with the lowest weighted cost per day among affordable stocks
        min_weighted_cost_row = affordable_df.loc[affordable_df['Weighted Cost per Day'].idxmin()]
        ticker = min_weighted_cost_row['Ticker']
        current_price = min_weighted_cost_row['Current Price']

        # Add 1 share to the allocation, but stop once the per-stock cap is reached
        shares_to_buy[ticker] += 1
        allocated_per_stock[ticker] += current_price
        remaining_amount -= current_price

        # Add the new purchase to the transactions for this ticker
        new_txn = pd.DataFrame([{
            'Activity/Trade Date': reference_date,
            'Transaction Date': reference_date,
            'Settlement Date': reference_date,
            'Activity Type': 'Bought',
            'Description': f'Weekly investment buy',
            'Symbol': ticker,
            'Cusip': '--',
            'Quantity #': 1,
            'Price $': current_price,
            'Amount $': -current_price,
            'Commission': 0.0,
            'Category': '--',
            'Note': '--'
        }])
        ticker_transactions[ticker] = pd.concat([ticker_transactions[ticker], new_txn], ignore_index=True)

        # Recalculate weighted cost per day from scratch using all transactions
        stock_txns = ticker_transactions[ticker].sort_values('Activity/Trade Date')

        total_buy_cost = 0
        total_sell_proceeds = 0
        total_shares = 0
        weighted_net_cost = 0

        if len(stock_txns) > 0:
            first_transaction_date = stock_txns['Activity/Trade Date'].min()
            days_ago_from_first_transaction = (reference_date - first_transaction_date).days

            if np.isclose(decay_rate, 0):
                stock_weight = 1.0
            else:
                stock_weight = 1 - np.exp(-decay_rate * days_ago_from_first_transaction)

        for _, txn in stock_txns.iterrows():
            activity = txn.get('Activity Type')
            quantity = txn['Quantity #']
            amount = txn['Amount $']

            # Handle NaN values
            if pd.isna(quantity) or pd.isna(amount):
                continue

            if _is_buy_transaction(activity, amount):
                total_buy_cost += abs(amount)
                weighted_net_cost += abs(amount) * stock_weight
                total_shares += quantity
            elif _is_sell_transaction(activity, amount):
                total_sell_proceeds += abs(amount)
                weighted_net_cost -= abs(amount) * stock_weight

        # Calculate net cost and weighted cost per day
        net_cost = total_buy_cost - total_sell_proceeds

        if len(stock_txns) > 0:
            first_date = stock_txns['Activity/Trade Date'].min()
            last_date = stock_txns['Activity/Trade Date'].max()
            total_days = (last_date - first_date).days

            if total_days > 0 and total_shares > 0:
                weighted_cost_per_day = weighted_net_cost / total_days
            else:
                weighted_cost_per_day = 0
        else:
            weighted_cost_per_day = 0

        # Update the DataFrame
        idx = cost_basis_df[cost_basis_df['Ticker'] == ticker].index[0]
        cost_basis_df.at[idx, 'Net Cost'] = net_cost
        cost_basis_df.at[idx, 'Total Shares'] = total_shares
        cost_basis_df.at[idx, 'Weighted Cost per Day'] = weighted_cost_per_day

    # Convert to DataFrame
    result_df = pd.DataFrame([
        {'Ticker': ticker, 'Shares to Buy': shares}
        for ticker, shares in shares_to_buy.items()
        if shares > 0
    ])

    # Return the updated cost_basis_df (which contains all stocks with their updated values)
    return result_df, cost_basis_df


def compute_effective_annual_rate(amounts, dates, current_value, current_date):
    """
    Solve for the annual effective rate of return given a series of signed cash flows.

    This matches the main branch formula:
    sum(amounts * (1 + r)^t) = current_value
    where time is measured in years from each transaction date to current_date.
    """
    amounts = -np.asarray(amounts, dtype=float)
    dates = pd.to_datetime(pd.Series(dates), format='mixed')
    durations = np.array([(pd.Timestamp(current_date) - d).days / 365.25 for d in dates])

    def residual(r_arr):
        r = r_arr[0]
        predicted = np.sum(amounts * np.power(np.maximum(1.0 + r, 1e-9), durations))
        return [predicted - current_value]

    res = least_squares(residual, x0=[0.1], bounds=(-0.999, 100.0))
    return float(res.x[0])


def compute_portfolio_effective_annual_return(transactions_df, current_value, current_date=None):
    """Compute the effective annual return for the portfolio using the main-branch equation."""
    if current_date is None:
        current_date = pd.Timestamp.now().normalize()

    return compute_effective_annual_rate(
        transactions_df['Amount $'].astype(float).to_numpy(),
        transactions_df['Activity/Trade Date'].tolist(),
        current_value,
        current_date,
    )


def get_decay_rate(t1, t2, difference):
    """
    Estimate the decay rate d based on the constraint:
    (exp(-d*t1) - exp(-d*t2))/(1-exp(-d*t1)) = difference

    Args:
        t1: Time in days for first point
        t2: Time in days for second point (t2 > t1)
        difference: The difference in weights between the two points

    Returns:
        float: The estimated decay rate d
    """
    def equation(d):
        """
        Equation to solve: (exp(-d*t1) - exp(-d*t2))/(1-exp(-d*t1)) = difference
        """
        exp_t1 = np.exp(-d * t1)
        exp_t2 = np.exp(-d * t2)
        return (exp_t1 - exp_t2) / (1 - exp_t1) - difference

    # Initial guess for decay rate (small positive value)
    initial_guess = 0.01

    # Solve for d
    d_solution = fsolve(equation, initial_guess)[0]

    return d_solution
