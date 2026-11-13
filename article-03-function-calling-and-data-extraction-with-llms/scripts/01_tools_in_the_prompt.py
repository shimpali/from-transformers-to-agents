# What a model actually reads when you give it tools
# pip install transformers
#
# Tool definitions are not sent through a separate channel. The chat template
# writes them into the system prompt as text, so they cost tokens on every request.
import textwrap
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def make_tool_definition(name, description, parameter, parameter_description):
    """One tool definition in the JSON shape chat APIs use."""
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {parameter: {"type": "string", "description": parameter_description}},
            "required": [parameter],
        },
    }}


tools = [
    make_tool_definition("fetch_arxiv", "Fetch the text of a paper from arXiv by its ID.", "arxiv_id", "The arXiv ID, for example 2408.02442"),
    make_tool_definition("get_section", "Return one named section of the paper already fetched.", "name", "Section title, for example Results"),
    make_tool_definition("search_arxiv", "Search arXiv and return titles with their IDs.", "query", "Search words"),
    make_tool_definition("list_sections", "List the section titles of the fetched paper.", "arxiv_id", "The arXiv ID"),
    make_tool_definition("get_references", "Return the reference list of the fetched paper.", "arxiv_id", "The arXiv ID"),
    make_tool_definition("run_python", "Run a short Python snippet and return its output.", "code", "Python source code"),
    make_tool_definition("read_file", "Read a text file from the lab folder.", "path", "Relative file path"),
    make_tool_definition("write_file", "Write a text file into the lab folder.", "path", "Relative file path"),
    make_tool_definition("list_files", "List the files in the lab folder.", "folder", "Relative folder path"),
    make_tool_definition("get_abstract", "Return only the abstract of a paper.", "arxiv_id", "The arXiv ID"),
]

messages = [{"role": "user", "content": "Get paper 2408.02442"}]


def count_prompt_tokens(tool_count):
    template_options = {"tools": tools[:tool_count]} if tool_count else {}
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **template_options)
    return len(tokenizer(text).input_ids), text


tokens_without_tools, _ = count_prompt_tokens(0)
print(f"{'tools':>5}  {'prompt tokens':>13}  {'added by the tools':>18}")
for tool_count in (0, 1, 2, 5, 10):
    token_count, _ = count_prompt_tokens(tool_count)
    print(f"{tool_count:>5}  {token_count:>13}  {token_count - tokens_without_tools:>18}")  # tokens beyond the no-tool prompt

# The system block the model really reads when one tool is given
_, text = count_prompt_tokens(1)
print()
for line in text.split("<|im_start|>user")[0].splitlines():  # wrapped so it fits a narrow page
    print(textwrap.fill(line, 76, subsequent_indent="  "))
