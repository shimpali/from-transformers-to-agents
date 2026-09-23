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
