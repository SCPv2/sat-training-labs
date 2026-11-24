import argparse
import json
import tempfile
from pathlib import Path

import torch
from accelerate import init_empty_weights
from datasets import Dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from train import attach_lora, encode, read_rows


parser = argparse.ArgumentParser()
parser.add_argument("--assets", type=Path, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.assets / "model", local_files_only=True)
lengths = []
split_prompts = []
for split in ["train", "validation", "test"]:
    rows = read_rows(args.assets / "dataset" / f"{split}.jsonl")
    split_prompts.append({" ".join(row["messages"][-2]["content"].lower().split()) for row in rows})
    lengths.extend(len(encode(tokenizer, row, 1536)["input_ids"]) for row in rows)
if any(split_prompts[first] & split_prompts[second] for first, second in [(0, 1), (0, 2), (1, 2)]):
    raise ValueError("Exact instruction overlap between splits")
config = AutoConfig.from_pretrained(args.assets / "model", local_files_only=True)
with init_empty_weights():
    model = attach_lora(AutoModelForCausalLM.from_config(config))
trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
if not trainable:
    raise ValueError("No trainable adapter parameters")
print(json.dumps({"rows": len(lengths), "max_tokens": max(lengths), "lora_parameters": trainable}), flush=True)

tiny = config.to_dict()
tiny["vision_config"] = None
tiny["audio_config"] = None
tiny["text_config"].update({"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                            "global_head_dim": 16, "vocab_size": 128, "vocab_size_per_layer_input": 128,
                            "hidden_size_per_layer_input": 4, "num_kv_shared_layers": 2,
                            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                            "max_position_embeddings": 128, "sliding_window": 32})
tiny_config = type(config).from_dict(tiny)
tiny_model = attach_lora(AutoModelForCausalLM.from_config(tiny_config))
tokens = torch.tensor([[2, 10, 11, 12, 13, 14, 15, 1]])
loss = tiny_model(input_ids=tokens, labels=tokens, use_cache=False).loss
loss.backward()
if not torch.isfinite(loss) or not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                      for parameter in tiny_model.parameters() if parameter.requires_grad):
    raise ValueError("CPU LoRA backward failed")
with tempfile.TemporaryDirectory() as directory:
    miniature_data = Dataset.from_list([{"input_ids": tokens[0].tolist(), "labels": tokens[0].tolist()}])
    trainer = Trainer(model=tiny_model, train_dataset=miniature_data, eval_dataset=miniature_data,
                      args=TrainingArguments(output_dir=directory + "/checkpoints", use_cpu=True, max_steps=1,
                                             per_device_train_batch_size=1, report_to="none",
                                             gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant": False},
                                             eval_strategy="epoch", save_strategy="epoch",
                                             load_best_model_at_end=True, label_names=["labels"]))
    trainer.train()
    tiny_model.save_pretrained(directory)
    restored = PeftModel.from_pretrained(AutoModelForCausalLM.from_config(tiny_config), directory)
    assert torch.isfinite(restored(input_ids=tokens, labels=tokens, use_cache=False).loss)
print("CPU miniature-model forward/backward and adapter reload passed", flush=True)
