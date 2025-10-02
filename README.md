This repo provides the code for this [blog post](https://weitz.blog/p/attention-explained-to-ordinary-programmers).

This repo has a .devcontainer directory, so opening it in VSCode should automatically
install most dependencies you need. The only thing you have to download manually are the Llama2 weights, by following these [instructions](https://github.com/meta-llama/llama-models?tab=readme-ov-file#download); tldr:

Go to: https://www.llama.com/llama2/
Click "Download the model"
Under "Previous language & safety models", check "Llama 2".
Run `llama download --source meta --model-id Llama-2-7b-chat`

To run the code, you need a machine with 64G of memory (no GPU required). 
If you don't have that, you can also open this repo in a github Codespace with 64G of memory.

Now run:

```
./src/example.py
```
