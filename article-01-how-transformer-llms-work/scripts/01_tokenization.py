# See how a real tokenizer splits a sentence into pieces + IDs
# pip install transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "My overjoyed squirrel stole peanuts."

for token_id in tokenizer(text)["input_ids"]:
    print(f"{token_id:>6}  ->  {tokenizer.decode([token_id])!r}")
