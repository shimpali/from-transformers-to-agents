# How Transformer LLMs Work -- all code from the article, concatenated
# Article 1 of the "From Transformers to Agents" series
# https://github.com/<your-username>/from-transformers-to-agents/tree/main/1-how-transformer-llms-work

# ---------------------------------------------------------------
# 1. Tokenization
# ---------------------------------------------------------------
# See how a real tokenizer splits a sentence into pieces + IDs
# pip install transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "My overjoyed squirrel stole peanuts."

for token_id in tokenizer(text)["input_ids"]:
    print(f"{token_id:>6}  ->  {tokenizer.decode([token_id])!r}")

# ---------------------------------------------------------------
# 2. Embeddings
# ---------------------------------------------------------------
# Cosine similarity between two words' embedding vectors
# pip install transformers torch
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
table = model.get_input_embeddings().weight

def vector_for(word):
    return table[tokenizer.convert_tokens_to_ids(word)]

for a, b in [("coffee", "espresso"), ("coffee", "guitar")]:
    sim = torch.cosine_similarity(vector_for(a), vector_for(b), dim=0)
    print(f"similarity({a!r}, {b!r}) = {sim.item():.3f}")

# ---------------------------------------------------------------
# 3. Self-attention
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 4. The generation loop
# ---------------------------------------------------------------
# The autoregressive loop underneath every chat response, made explicit
# pip install transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
input_ids = tokenizer("This transformer block is composed of", return_tensors="pt").input_ids

for _ in range(8):
    with torch.no_grad():
        logits = model(input_ids).logits        # forward pass through every block
    next_id = logits[0, -1].argmax()             # greedy: pick highest-probability token
    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)  # append, then repeat

print(tokenizer.decode(input_ids[0]))
