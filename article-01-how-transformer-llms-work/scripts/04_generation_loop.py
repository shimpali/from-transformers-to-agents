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
