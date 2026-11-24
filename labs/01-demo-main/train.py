import argparse
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from settings import configure_environment

configure_environment()

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, set_seed)

from storage import download, upload


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Empty dataset: " + str(path))
    return rows


def render(tokenizer, messages, generation=False):
    return tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=generation, enable_thinking=False, return_dict=False)


def encode(tokenizer, row, max_length):
    messages = row["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("Training row must end with assistant response")
    prompt = render(tokenizer, messages[:-1], generation=True)
    complete = render(tokenizer, messages)
    if complete[:len(prompt)] != prompt:
        raise ValueError("Chat template prompt mismatch; refusing incorrect loss masking")
    if len(complete) > max_length:
        raise ValueError(f"Sequence has {len(complete)} tokens; increase MAX_LENGTH={max_length}")
    labels = [-100] * len(prompt) + complete[len(prompt):]
    if not any(label != -100 for label in labels):
        raise ValueError("No assistant tokens to train")
    return {"input_ids": complete, "attention_mask": [1] * len(complete), "labels": labels}


def attach_lora(model):
    targets = [name for name, module in model.named_modules()
               if "language_model" in name and name.endswith(("q_proj", "v_proj"))
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No supported language-model attention projections found")
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=targets, task_type="CAUSAL_LM", bias="none"))


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda")


def evaluate(model, tokenizer, rows, allowed, output, max_new_tokens):
    model.eval()
    predictions = []
    expected_intents, predicted_intents = [], []
    valid = correct = 0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            tokens = render(tokenizer, row["messages"][:-1], generation=True)
            input_ids = torch.tensor([tokens], device="cuda")
            generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                       max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id, use_cache=True)
            text = tokenizer.decode(generated[0, len(tokens):], skip_special_tokens=True).strip()
            expected = json.loads(row["messages"][-1]["content"])
            try:
                parsed = json.loads(text)
                schema_valid = isinstance(parsed, dict) and set(parsed) == {"category", "intent"}
                schema_valid = schema_valid and all(isinstance(value, str) for value in parsed.values())
            except (ValueError, TypeError):
                parsed, schema_valid = {}, False
            label_valid = schema_valid and (parsed["category"], parsed["intent"]) in allowed
            valid += int(schema_valid)
            correct += int(parsed == expected)
            expected_intents.append(expected["intent"])
            predicted_intents.append(parsed["intent"] if label_valid else "__invalid__")
            predictions.append({"input": row["messages"][-2]["content"], "expected": expected, "prediction": text})
            if (index + 1) % 25 == 0:
                print(f"Evaluation {index + 1}/{len(rows)}", flush=True)
    metrics = {"samples": len(rows), "intent_accuracy": accuracy_score(expected_intents, predicted_intents),
               "macro_f1": f1_score(expected_intents, predicted_intents, labels=sorted({intent for _, intent in allowed}),
                                    average="macro", zero_division=0),
               "json_schema_rate": valid / len(rows), "exact_label_accuracy": correct / len(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    output.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    print(json.dumps(metrics), flush=True)


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required; check driver and container runtime")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This demo supports one process / one GPU")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU to this container")
    set_seed(42)
    root = Path(os.environ.get("WORK_DIR", "/work"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / "runs" / run_id
    output.mkdir(parents=True)
    output_uri = os.environ["OUTPUT_URI"].rstrip("/") + "/" + run_id
    failure = None
    try:
        manifest = download(os.environ["INPUT_URI"], root / "input")
        (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
        (output / "run-config.json").write_text(json.dumps({**vars(args), "run_id": run_id,
            "input_uri": os.environ["INPUT_URI"], "output_uri": output_uri, "gpu": torch.cuda.get_device_name()}, indent=2))
        shutil.copyfile("/opt/requirements-resolved.txt", output / "requirements-resolved.txt")
        model_dir = root / "input/model"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_rows = read_rows(root / "input/dataset/train.jsonl")
        validation_rows = read_rows(root / "input/dataset/validation.jsonl")
        test_rows = read_rows(root / "input/dataset/test.jsonl")
        if args.eval_samples:
            test_rows = test_rows[:args.eval_samples]
        allowed = {(row["category"], row["intent"]) for row in json.loads((root / "input/dataset/labels.json").read_text())}
        train_dataset = Dataset.from_list([encode(tokenizer, row, args.max_length) for row in train_rows])
        validation_dataset = Dataset.from_list([encode(tokenizer, row, args.max_length) for row in validation_rows])
        model = load_model(model_dir)
        evaluate(model, tokenizer, test_rows, allowed, output / "metrics/baseline", args.max_new_tokens)
        model = attach_lora(model)
        model.config.get_text_config().use_cache = False
        model.print_trainable_parameters()
        trainer = Trainer(model=model, processing_class=tokenizer,
                          train_dataset=train_dataset, eval_dataset=validation_dataset,
                          data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
                          args=TrainingArguments(output_dir=str(output / "checkpoints"),
                              num_train_epochs=args.epochs, max_steps=args.max_steps,
                              per_device_train_batch_size=1, per_device_eval_batch_size=1,
                              gradient_accumulation_steps=8, learning_rate=1e-4, bf16=True,
                              gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                              eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                              load_best_model_at_end=True, metric_for_best_model="eval_loss",
                              greater_is_better=False, logging_steps=5, report_to="none",
                              seed=42, data_seed=42, label_names=["labels"]))
        trainer.train()
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        (output / "training-log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        del trainer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(load_model(model_dir), output / "adapter", is_trainable=False)
        evaluate(model, tokenizer, test_rows, allowed, output / "metrics/finetuned", args.max_new_tokens)
        (output / "status.json").write_text(json.dumps({"training": "completed"}))
    except Exception as error:
        failure = error
        traceback.print_exc()
        (output / "error.txt").write_text(traceback.format_exc())
        (output / "status.json").write_text(json.dumps({"training": "failed"}))
    try:
        upload(output_uri, output)
    except Exception:
        traceback.print_exc()
        if failure is not None:
            raise failure
        raise
    if failure is not None:
        raise failure
    print("Training and output upload completed:", output_uri, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU Gemma 4 LoRA training from S3 inputs")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--eval-samples", type=int, default=0, help="0 evaluates the full test split")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    arguments = parser.parse_args()
    if arguments.max_length < 1 or arguments.max_new_tokens < 1 or arguments.eval_samples < 0:
        parser.error("Invalid length or sample count")
    run(arguments)
