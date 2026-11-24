#!/usr/bin/env python3
"""Gemma 4 E2B-it QLoRA experiment on Korean news or clothing images."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from checkpointing import S3CheckpointCallback, prepare_resume, training_fingerprint
from settings import storage_uri
from storage import upload_results
from transformers import (AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
    Gemma4AudioFeatureExtractor, Gemma4ImageProcessor, Gemma4VideoProcessor, Gemma4Processor,
    Trainer, TrainingArguments)


def split(root, task, name):
    directory = root / task
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / (name + ('.jsonl' if task == 'text' else '.npz'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['files'][name]['sha256']
    if task == 'text':
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        labels = manifest['labels']
    else:
        with np.load(path) as data:
            images, targets = data['images'].copy(), data['labels'].copy()
        rows = [{'id': name + '-' + str(i), 'image': image, 'label': int(target)}
                for i, (image, target) in enumerate(zip(images, targets))]
        labels = manifest['labels_ko']
    assert len(rows) == manifest['files'][name]['count']
    return rows, labels, digest


def balanced_subset(rows, class_count, limit):
    buckets = [[row for row in rows if row['label'] == label] for label in range(class_count)]
    selected = []
    while len(selected) < min(limit, len(rows)):
        added = False
        for bucket in buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
                added = True
        if not added:
            break
    return selected


def picture(row):
    return Image.fromarray(row['image']).resize((224, 224), Image.Resampling.NEAREST).convert('RGB')


def messages(row, task, labels, answer=False):
    choices = ', '.join(labels)
    if task == 'text':
        query = '한국어 뉴스 제목의 주제를 하나만 고르세요. 선택지: ' + choices + '. 주제 이름만 답하세요. 제목: ' + row['title']
        content = [{'type': 'text', 'text': query}]
    else:
        query = '이미지의 의류 종류를 하나만 고르세요. 선택지: ' + choices + '. 종류 이름만 답하세요.'
        content = [{'type': 'image', 'image': picture(row)}, {'type': 'text', 'text': query}]
    items = [{'role': 'user', 'content': content}]
    if answer:
        items.append({'role': 'assistant', 'content': [{'type': 'text', 'text': labels[row['label']]}]})
    return items


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


def collate(processor, task, labels):
    def inner(examples):
        assert len(examples) == 1
        row = examples[0]
        prompt = processor.apply_chat_template(messages(row, task, labels), tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(messages(row, task, labels, True), tokenize=False, add_generation_prompt=False)
        images = [picture(row)] if task == 'image' else None
        prompt_ids = processor(text=[prompt], images=images, return_tensors='pt')['input_ids'][0]
        batch = processor(text=[full], images=images, return_tensors='pt')
        ids = batch['input_ids'][0]
        if not torch.equal(ids[:len(prompt_ids)], prompt_ids):
            raise ValueError('Prompt tokens are not prefix of completion; cannot mask loss safely')
        target = ids.clone()
        target[:len(prompt_ids)] = -100
        assert torch.any(target != -100)
        batch['labels'] = target.unsqueeze(0)
        return batch
    return inner


def evaluate(model, processor, rows, task, labels, destination):
    gold, pred, details = [], [], []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            prompt = processor.apply_chat_template(messages(row, task, labels), tokenize=False, add_generation_prompt=True)
            images = [picture(row)] if task == 'image' else None
            batch = processor(text=[prompt], images=images, return_tensors='pt').to('cuda')
            result = model.generate(**batch, max_new_tokens=24, do_sample=False, disable_compile=True,
                                    pad_token_id=processor.tokenizer.pad_token_id)
            raw = processor.batch_decode(result[:, batch['input_ids'].shape[-1]:], skip_special_tokens=True)[0].strip()
            label = raw.splitlines()[0].strip().rstrip('.。') if raw else ''
            if label not in labels:
                label = '__invalid__'
            gold.append(row['label'])
            pred.append(labels.index(label) if label in labels else -1)
            details.append({'id': row['id'], 'input': row.get('title', '[image]'),
                            'label': labels[row['label']], 'prediction': label, 'raw_output': raw})
            if index % 10 == 0:
                print(json.dumps({'phase': destination.stem, 'evaluated': index, 'total': len(rows)}), flush=True)
    metrics = {'count': len(rows), 'accuracy': accuracy_score(gold, pred),
               'macro_f1': f1_score(gold, pred, labels=list(range(len(labels))), average='macro', zero_division=0),
               'valid_label_rate': sum(value >= 0 for value in pred) / len(pred)}
    destination.write_text(json.dumps({'metrics': metrics, 'predictions': details}, ensure_ascii=False, indent=2))
    return metrics


def load_processor(model_dir):
    # The supplied offline manifest includes processor_config.json with embedded
    # image/audio/video configs, but no separate preprocessor_config.json.
    config = json.loads((model_dir / 'processor_config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(**config['feature_extractor']),
        image_processor=Gemma4ImageProcessor(**config['image_processor']),
        video_processor=Gemma4VideoProcessor(**config['video_processor']),
        tokenizer=tokenizer,
        chat_template=(model_dir / 'chat_template.jinja').read_text(),
        image_seq_length=config['image_seq_length'],
        audio_seq_length=config['audio_seq_length'],
        audio_ms_per_token=config['audio_ms_per_token'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['text', 'image'], required=True)
    parser.add_argument('--bundle', type=Path, default=Path('/bundle'))
    parser.add_argument('--model', type=Path, default=Path('/model'))
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--train-count', type=int, default=700)
    parser.add_argument('--eval-count', type=int, default=50)
    parser.add_argument('--checkpoint-steps', type=int, default=20)
    parser.add_argument('--checkpoint-uri', default=None)
    parser.add_argument('--output-uri', default=None)
    args = parser.parse_args()
    assert args.steps > 0 and args.eval_count > 0 and args.checkpoint_steps > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    random.seed(42)
    torch.manual_seed(42)
    started = time.time()
    out = args.output / args.task
    out.mkdir(parents=True, exist_ok=True)
    train, labels, train_hash = split(args.bundle, args.task, 'train')
    test, test_labels, test_hash = split(args.bundle, args.task, 'test')
    assert labels == test_labels
    if args.task == 'image':
        train = balanced_subset(train, len(labels), args.train_count)
        test = balanced_subset(test, len(labels), args.eval_count)
    else:
        train, test = train[:args.train_count], test[:args.eval_count]
    checkpoint_uri = storage_uri(args.checkpoint_uri, 'SAT_CHECKPOINT_URI')
    output_uri = storage_uri(args.output_uri, 'SAT_OUTPUT_URI')
    if checkpoint_uri and output_uri and checkpoint_uri.rstrip('/') == output_uri.rstrip('/'):
        raise ValueError('Checkpoint and result S3 prefixes must differ')
    fingerprint = training_fingerprint(model='google/gemma-4-E2B-it', task=args.task,
        train_sha256=train_hash, train_count=len(train), steps=args.steps,
        recipe='qlora-r8-bf16-lr2e-4-batch1-accum4-v1')
    checkpoint, store = prepare_resume(out / 'checkpoints', checkpoint_uri, fingerprint)
    if checkpoint:
        print(json.dumps({'resuming_from': checkpoint.name, 'source': 's3' if store else 'local'}), flush=True)
    processor = load_processor(args.model)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, quantization_config=quant, device_map='cuda:0', attn_implementation='eager')
    print('SAT_PHASE baseline_eval', flush=True)
    before = evaluate(model, processor, test, args.task, labels, out / 'before.json')
    model = prepare_model_for_kbit_training(model)
    targets = [name for name, layer in model.named_modules()
               if 'language_model' in name and name.endswith(('q_proj', 'v_proj'))
               and isinstance(layer, torch.nn.Linear)]
    if not targets:
        raise ValueError('Gemma language attention projections not found')
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                            target_modules=targets, task_type='CAUSAL_LM'))
    model.config.get_text_config().use_cache = False
    trainer = Trainer(model=model, train_dataset=Rows(train),
        data_collator=collate(processor, args.task, labels),
        args=TrainingArguments(output_dir=str(out / 'checkpoints'), max_steps=args.steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4,
            bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=5,
            save_strategy='steps', save_steps=min(args.checkpoint_steps, args.steps),
            save_total_limit=2, save_only_model=False, report_to='none',
            remove_unused_columns=False, label_names=['labels']),
        callbacks=[S3CheckpointCallback(store, fingerprint)] if store else None)
    print('SAT_PHASE training', flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(out / 'adapter'))
    processor.save_pretrained(out / 'adapter')
    (out / 'training-log.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    model.config.get_text_config().use_cache = True
    print('SAT_PHASE final_eval', flush=True)
    after = evaluate(model, processor, test, args.task, labels, out / 'after.json')
    summary = {'task': args.task, 'model': 'google/gemma-4-E2B-it', 'method': 'QLoRA',
        'steps': args.steps, 'train_count': len(train), 'test_count': len(test),
        'train_sha256': train_hash, 'test_sha256': test_hash,
        'before': before, 'after': after, 'accuracy_delta_pp': 100 * (after['accuracy'] - before['accuracy']),
        'gpu': torch.cuda.get_device_name(0), 'elapsed_seconds': time.time() - started,
        'resumed_from_step': int(checkpoint.name.rsplit('-', 1)[1]) if checkpoint else None,
        'checkpoint_uri': checkpoint_uri, 'output_uri': output_uri}
    (out / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if output_uri:
        uploaded = upload_results(output_uri, out)
        print(json.dumps({'result_files_uploaded': len(uploaded), 'output_uri': output_uri}), flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
