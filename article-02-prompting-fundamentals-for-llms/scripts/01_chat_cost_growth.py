# How much text the model reads on every turn of a chat
# pip install transformers
#
# A chat model has no memory between messages, so each request sends the
# whole conversation again. This builds a chat turn by turn and counts the
# tokens in every request with a real chat model's tokenizer.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

user_turns = [
    "Can you help me plan a small book club?",
    "We meet once a month, about eight people.",
    "Suggest three novels under 300 pages.",
    "Why did you pick the second one?",
    "Give me five discussion questions for it.",
    "Make question three less academic.",
    "Draft a short invite message for the group.",
    "thanks",
]
# Stand-in replies of a realistic length, so the history grows like a real chat
reply = "Here is a detailed answer with a few suggestions and the reasoning behind them. " * 6

history, total = [], 0
# sent = tokens in this request, total = tokens sent so far
print(f"{'turn':>4}  {'message':<44} {'sent':>6}  {'total':>6}")
for i, text in enumerate(user_turns, start=1):
    history.append({"role": "user", "content": text})
    # Exactly what gets sent: every earlier message plus the new one
    sent = len(tokenizer.apply_chat_template(history, add_generation_prompt=True, return_dict=True)["input_ids"])
    total += sent
    print(f"{i:>4}  {text:<44} {sent:>6}  {total:>6}")
    history.append({"role": "assistant", "content": reply})

thanks_alone = len(tokenizer("thanks").input_ids)
print(f"\n'thanks' alone is {thanks_alone} token; as turn {len(user_turns)} it sends {sent}.")
