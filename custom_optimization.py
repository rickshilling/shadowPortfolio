import jax
import jax.numpy as jnp
import optax
import numpy as np

def set_weights(shadow_transactions):
    num_stocks = shadow_transactions['num_stocks']
    weights = np.ones((num_stocks,1))
    # penalty_indices = []
    # for stock_index in range(num_stocks):
    #     note = shadow_transactions['Notes'][stock_index]
    #     if ("Earnings probation" not in note) and ("Exceeds" not in note):
    #         weights[stock_index] = 1/shadow_transactions['current_prices'][stock_index]
    #         # weights[stock_index] = 1/shadow_transactions['Rel PriceStrgth(%)'][stock_index]
    #     else:
    #         penalty_indices.append(stock_index)
    # penalty_weight = 10*jnp.max(weights)
    # # a < b, var(acx) = var(bdx) => 
    # #           (ac)^2var(x)= (bd)^2var(x)
    # #           a/b < 1 = ac^2/bd^2 
    # #           a/b < ac^2/bd^2
    # #           1 < c^2/d^2 
    # #           d^2 < c^2 
    # #           d < c 
    # #           c > d
    # for stock_index in penalty_indices:
    #     weights[stock_index] = penalty_weight
    shadow_transactions['weights'] = weights
    return shadow_transactions

def arg_min_variance(shadow_transactions, T=1, limit = 27*7, num_iterations = 1000):
    shadow_transactions = set_weights(shadow_transactions)
    # k[i] = new number of shares for stock i
    x = {"a": jnp.array(shadow_transactions['average_price_per_time']), 
         "n": jnp.array(shadow_transactions['num_transactions']),
         "u": jnp.array(shadow_transactions['current_prices']),
         "m": shadow_transactions['num_stocks'],
         "l": limit,
         "T": T,
         "w": jnp.array(shadow_transactions['weights'])}
    max_k = jnp.array(jnp.ceil(limit/x["u"]))
    # max_k = jnp.zeros_like(x["u"])
    params = {"k":max_k}
    start_learning_rate = 1e-2
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)
    losses = []
    amounts = []
    for iteration in range(num_iterations):
        current_loss, grads = jax.value_and_grad(loss)(params, x)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        amount = jnp.dot(jnp.squeeze(params["k"]),jnp.squeeze(x["u"]))
        losses.append(current_loss)
        amounts.append(amount)
        if jnp.mod(iteration, 1000)==0:
            float_list = jnp.squeeze(params["k"]).tolist()
            int_parameters = [int(x) for x in float_list]
            print((current_loss.item(), int_parameters))
    float_list = jnp.squeeze(params["k"]).tolist()
    int_parameters = [int(x) for x in float_list]
    print("Learned parameters:", float_list)
    new_average_price_per_time = model(params,x)
    shadow_transactions["new_shares"] = params["k"]
    shadow_transactions["new_average_price_per_time"] = new_average_price_per_time
    shadow_transactions["new_amount"] = amount
    return shadow_transactions, losses
    
def model(params, x):
    a=x["a"]
    n=x["n"]
    u=x["u"]
    m=x["m"]
    T=x["T"]
    w=x["w"]
    k=params["k"]
    threshold = 0.5
    positive_indices = jnp.where(k>=threshold)
    negative_indices = jnp.where(k<threshold)
    ap = a[positive_indices]
    np = n[positive_indices]
    kp = k[positive_indices]
    up = u[positive_indices]
    np = n[positive_indices]
    wp = w[positive_indices]

    positive_averages = (ap*np + wp*kp*up/T)/(np+1)
    # positive_averages = (ap*np + kp*up/T)/(np+1)
    negative_averages = a[negative_indices]
    averages = jnp.zeros_like(a)
    averages = averages.at[positive_indices].set(positive_averages)
    averages = averages.at[negative_indices].set(negative_averages)
    # averages = (a*n + k*u/T)/(n+1)
    return averages

def loss(params, x):
    averages = model(params, x)
    k=jnp.squeeze(params["k"])
    l=jnp.squeeze(x["l"])
    u=jnp.squeeze(x["u"])
    
    # penalty_factor = 1.0e5
    # negative_penalty = penalty_factor * jnp.mean(jnp.maximum(0, -k)**2)

    tau = 1e-1
    negative_penalty = jnp.exp(-k/tau)

    loss = jnp.var(averages) + (l - jnp.dot(k,u))**2 + jnp.sum(negative_penalty)
    # mean_averages = jnp.mean(averages)
    # residual = averages - mean_averages
    # loss =  jnp.mean((residual)**2) + (l - jnp.dot(k,u))**2 + negative_penalty
    return loss
