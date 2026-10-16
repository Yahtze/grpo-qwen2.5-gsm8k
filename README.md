# GRPO on Qwen2.5 with GSM8K

Notebooks for evaluating and fine-tuning [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on [GSM8K](https://huggingface.co/datasets/openai/gsm8k), a dataset of grade-school math problems. Training uses Group Relative Policy Optimization (GRPO) with LoRA adapters and a 4-bit quantized model.

## Notebooks

- **[Baseline evaluation](notebooks/qwen25_gsm8k_baseline.ipynb):** Evaluates the model on the GSM8K test split with greedy decoding. Reports numeric-answer accuracy, parse failures, and inference speed.
- **[GRPO training](notebooks/qwen25_gsm8k_grpo_lora_sanity.ipynb):** Runs a short 16-step experiment on 64 training examples, then evaluates 64 held-out examples with eight completions per problem. Rewards numeric correctness and answer formatting.

## Getting started

1. Open a notebook in Google Colab or Jupyter with Python 3.11+ and a CUDA-enabled PyTorch runtime. A GPU is required; the notebooks target a Colab T4.
2. Before running the imports, install the notebook dependencies in a code cell:

   ```python
   %pip install transformers datasets accelerate bitsandbytes huggingface_hub tqdm numpy trl peft
   ```

3. Review the settings near the top of the notebook, then run the remaining cells in order. Model and dataset downloads require internet access on the first run.

For a quick baseline check, set `MAX_EXAMPLES` to a small number and reduce `BATCH_SIZE` if GPU memory is limited.

## Outputs

Both notebooks display metrics and sample answers in the notebook. The baseline uses the official test split; the training notebook uses a separate holdout from the training split. Their scores use different data and decoding settings, so they are not a direct before-and-after comparison.

The training notebook does not save model checkpoints. Save the executed notebook if you want to retain its output.
