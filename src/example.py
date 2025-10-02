# from llmkv import autocomplete
from llm import autocomplete
from llama.model import llama
from llama.tokenizer import Tokenizer
from llama.params import llama7BParams, llama1BParams
import sys

max_seq_len = 100

tokenizer = Tokenizer(
    model_path="/home/vscode/.llama/checkpoints/Llama-2-7b-chat/tokenizer.model")
transformer = llama(llama7BParams(), max_seq_len)
prompt = sys.argv[1]

tokens = tokenizer.encode(prompt, bos=True, eos=False)
for token in autocomplete(transformer, max_seq_len, tokens):
    if (token == tokenizer.eos_id):
        break
    print(tokenizer.decode([token]), end=' ', flush=True)
