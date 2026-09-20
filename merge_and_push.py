# merge_and_push.py
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "./results",
    device_map="cpu",
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model", safe_serialization=True, max_shard_size="2GB")

tokenizer = AutoTokenizer.from_pretrained("./results")
tokenizer.save_pretrained("./merged-model")