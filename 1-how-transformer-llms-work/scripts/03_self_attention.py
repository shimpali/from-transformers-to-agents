# Self-attention from scratch on toy vectors -- numpy only,
# so the two steps stay visible instead of hiding in a framework call.
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

tokens = ["Maya", "walked", "the", "dog"]
X = np.array([[1,0,1,0], [0,1,0,1], [1,1,0,0], [0,0,1,1]], dtype=float)

np.random.seed(0)  # untrained, random projections -- illustrates the mechanism only
W_query, W_key, W_value = (np.random.rand(4, 4) for _ in range(3))
queries, keys, values = X @ W_query, X @ W_key, X @ W_value

current = -1  # "dog"
weights = softmax(queries[current] @ keys.T)     # step 1: relevance scoring
context_vector = weights @ values                 # step 2: combine information

for token, weight in zip(tokens, weights):
    print(f"{token:>6}: {weight:.2%} relevant to 'dog'")
