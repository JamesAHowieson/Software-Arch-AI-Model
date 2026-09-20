# Software Architecture Model

This project fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` on the `ajibawa-2023/Software-Architecture` dataset using 4-bit QLoRA. This built this model hosted on [HuggingFace](https://huggingface.co/JamesAHowieson/SoftwareArchitecture)

## Requirements

- Python 3.11 or 3.12 is recommended.
- NVIDIA GPU with CUDA support for practical local training.
- Enough disk space for the base model, dataset, and checkpoints.
- A Hugging Face account if downloading private resources or uploading results.

## Authentication

Set the token in the environment before starting the notebook, or enter it when the configuration cell prompts for it:

```bash
export HF_TOKEN="hf_your_token_here"
```

Never commit a token to the notebook or repository. Use a Hugging Face token with read access for downloads and write access for uploads.

## Notebook Workflow

Open `SoftwareArchModel.ipynb` and run the cells in order:

1. Install the Python and CUDA dependencies.
2. Configure Hugging Face access.
3. Run the optional Transformers sanity check.
4. Load the quantized Qwen model and attach LoRA adapters.
5. Download and split the software-architecture JSONL dataset.
6. Format and tokenize the examples for causal language-model training.
7. Train the adapter with `Trainer`.
8. Evaluate the adapter and upload it if required.
9. Run the final inference smoke test.

Restart the notebook kernel before rerunning the model-loading cell after changing dependencies or model configuration.

## Training Output

Training saves the reusable adapter and tokenizer under:

```text
./results
```

The results are LoRA adapter weights, not a standalone copy of the base model. Checkpoint saving is enabled so interrupted runs can be resumed or recovered.

## Load The Adapter

Use the base model together with the saved adapter in another application:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_path = "./results"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

Replace `adapter_path` with the Hugging Face repository ID after uploading the adapter.

## Upload To Hugging Face

Authenticate with the Hugging Face CLI:

```bash
hf auth login
```

The final notebook upload cell uploads `./results` to the configured model repository. Uploading the adapter is much smaller and safer than merging the full model locally.

To create a standalone model containing merged weights, use a separate machine with substantially more RAM and disk. Do not run `merge_and_unload()` in the training notebook if it causes the local environment to run out of memory.
