This repo provides the code for this [blog post](https://weitz.blog/p/attention-explained-to-ordinary-programmers). The code that defines the LLM architecture, including the `attend_to` function, is in [src/llm.py](src/llm.py).

This repo defines a devcontainer, so opening it in VSCode should automatically
install most dependencies you need. The only thing you have to download manually are the Llama2 weights, by following these [instructions](https://github.com/meta-llama/llama-models?tab=readme-ov-file#download). Tldr:

Go to https://www.llama.com/llama2/.
Click "Download the model".
Under "Previous language & safety models", check "Llama 2".
Then run `llama download --source meta --model-id Llama-2-7b-chat`.

To run the code, you need a machine with 64G of memory (no GPU required).
If you don't have that, you can also open this repo in a github Codespace with 64G of memory.

To run the LLM execute:

```
python src/main.py "[INST] Always answer with Haiku. I am going to Paris, what should I see? [/INST]"
```

Which should print something like the following (at a rate of about 1 token every couple of seconds):

```
[ INST ] Always answer with Ha iku . I am going to Paris , what should I see ? [ / INST ]

Tra vel er ' s delight
R om ance in the city lights
E iff el Tower high
```

If you're impatient, you can also run:

```
python src/main.py "[INST] What is 1+1? [/INST]"
```

Which produces something like this within about a minute:

```
[ INST ] What is  1 + 1 ? [ / INST ]  The answer to  1 + 1 is  2 .
```
