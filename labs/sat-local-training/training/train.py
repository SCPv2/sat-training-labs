"""Offline, single-process PyTorch trainer for CPU and one CUDA GPU.

Run from project root:
  python train.py --task synthetic_images --epochs 3 --run-dir ./runtime/cli-cnn
  python train.py --task csv --csv-file samples/dataset.csv --epochs 3 --run-dir ./runtime/cli-csv
  python train.py --task csv --csv-file samples/dataset.csv --epochs 5 --run-dir ./runtime/cli-csv --resume

Checkpointing is synchronous at the end of each completed epoch (not concurrent checkpointing).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from training.common import make_images, make_model

should_stop = False


def request_stop(*_args):
    global should_stop
    should_stop = True


def load_csv(path):
    if not path or not Path(path).is_file():
        raise ValueError('CSV dataset missing; upload a dataset.csv to the Volume or use --csv-file')
    if Path(path).stat().st_size > 10 * 1024 * 1024:
        raise ValueError('CSV exceeds 10MiB limit')
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2 or len(header) > 65:
            raise ValueError('CSV requires numeric features and final numeric label (2..65 columns)')
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Inconsistent CSV column counts')
            try:
                rows.append([float(z) for z in row])
            except ValueError as exc:
                raise ValueError('CSV features and labels must be numeric') from exc
            if len(rows) > 50000:
                raise ValueError('CSV limited to 50,000 rows')
    if len(rows) < 20:
        raise ValueError('CSV needs at least 20 rows')
    t = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(t).all():
        raise ValueError('CSV has NaN or Infinity')
    labels, y = torch.unique(t[:, -1], sorted=True, return_inverse=True)
    if not (2 <= len(labels) <= 32):
        raise ValueError('CSV requires 2..32 distinct numeric classes')
    permutation = torch.randperm(len(t), generator=torch.Generator().manual_seed(42))
    train_idx = permutation[:int(len(t) * .8)]
    val_idx = permutation[int(len(t) * .8):]
    raw_x = t[:, :-1]
    mean = raw_x[train_idx].mean(0)
    std = raw_x[train_idx].std(0).clamp_min(1e-6)
    x = (raw_x - mean) / std
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), {
        'input_dim': int(x.shape[1]), 'classes': int(len(labels)), 'label_values': labels.tolist(),
        'feature_names': header[:-1], 'mean': mean.tolist(), 'std': std.tolist(),
    }


def cpu_copy(obj):
    """Convert checkpoint tensors to CPU without moving the live model/optimizer."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: cpu_copy(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu_copy(x) for x in obj)
    return obj


def atomic_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    global should_stop
    should_stop = False
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--task', choices=['synthetic_images', 'csv'], default='synthetic_images')
    p.add_argument('--csv-file', default='')
    p.add_argument('--epochs', type=int, default=5, help='Total target epoch; --resume continues up to this total')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--gpu-fraction', type=float, default=.7)
    p.add_argument('--precision', choices=['fp32', 'amp'], default='fp32')
    p.add_argument('--resume', action='store_true', help='Restore model, optimizer and epoch from last.pt')
    a = p.parse_args(argv)
    if not (1 <= a.epochs <= 50 and 4 <= a.batch_size <= 128 and .5 <= a.gpu_fraction <= .85):
        raise ValueError('epoch 1..50, batch 4..128, GPU fraction .5..85')
    if a.task == 'csv' and not a.csv_file:
        raise ValueError('--csv-file is required for CSV task')
    out = Path(a.run_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    cuda = torch.cuda.is_available()
    if a.device == 'cuda' and not cuda:
        raise RuntimeError('CUDA unavailable. Check NVIDIA driver and CUDA-enabled PyTorch installation')
    device = torch.device('cuda' if cuda and a.device != 'cpu' else 'cpu')
    if a.precision == 'amp' and device.type != 'cuda':
        raise ValueError('AMP fp16 requires CUDA; choose fp32 on CPU')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, device=0)
    print(f'Device={device}, PyTorch={torch.__version__}, precision={a.precision}', flush=True)
    if a.task == 'csv':
        train, val, details = load_csv(a.csv_file)
        fingerprint = hashlib.sha256(Path(a.csv_file).read_bytes()).hexdigest()
    else:
        train, val = make_images(384, 42), make_images(96, 128)
        details = {'input_dim': 0, 'classes': 3, 'label_values': [0, 1, 2],
                   'class_names': ['vertical', 'horizontal', 'diagonal']}
        fingerprint = 'synthetic-images-v1-seeds-42-128'
    model = make_model(a.task, int(details['input_dim']), int(details['classes'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    criterion = torch.nn.CrossEntropyLoss()
    amp = a.precision == 'amp'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    epoch_start = 1
    best_accuracy = -1.0
    last_path = out / 'last.pt'
    if a.resume:
        if not last_path.exists():
            raise FileNotFoundError('No last.pt checkpoint to resume')
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 2 or 'optimizer_state' not in checkpoint:
            raise ValueError('This checkpoint does not contain v2 optimizer state; resume impossible')
        if checkpoint['task'] != a.task or checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Training task or dataset changed since checkpoint was saved')
        if int(checkpoint['batch_size']) != a.batch_size or checkpoint['precision'] != a.precision:
            raise ValueError('Batch size / precision changed; resume requires matching training configuration')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if amp and checkpoint.get('scaler_state'):
            scaler.load_state_dict(checkpoint['scaler_state'])
        if checkpoint.get('rng_state') is not None:
            torch.set_rng_state(checkpoint['rng_state'])
        epoch_start = int(checkpoint['epoch']) + 1
        best_accuracy = float(checkpoint['best_accuracy'])
        if epoch_start > a.epochs:
            raise ValueError('Checkpoint already meets target epochs; increase --epochs to continue')
        print(f'Resumed from epoch={epoch_start-1}, target={a.epochs}', flush=True)
    else:
        # Direct CLI fresh starts only; API separately clears old run artifacts.
        for old in ('metrics.jsonl', 'best.pt', 'last.pt', 'run_state.json'):
            (out / old).unlink(missing_ok=True)
    train_loader = DataLoader(TensorDataset(*train), batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*val), batch_size=128, num_workers=0)
    start = time.monotonic()
    with (out / 'metrics.jsonl').open('a' if a.resume else 'w', encoding='utf-8', buffering=1) as metrics:
        for epoch in range(epoch_start, a.epochs + 1):
            if should_stop:
                break
            model.train()
            total, total_loss = 0, 0.0
            for xb, yb in train_loader:
                if should_stop:
                    break
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16 if amp else torch.bfloat16):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(yb)
                total += len(yb)
            if should_stop:
                break
            model.eval()
            count, correct, val_loss = 0, 0, 0.
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16 if amp else torch.bfloat16):
                        output = model(xb)
                        val_loss += criterion(output, yb).item() * len(yb)
                    correct += (output.argmax(1) == yb).sum().item()
                    count += len(yb)
            accuracy = correct / max(1, count)
            metric = {'epoch': epoch, 'train_loss': round(total_loss/max(1, total), 5),
                      'val_loss': round(val_loss/max(1, count), 5), 'accuracy': round(accuracy, 5),
                      'seconds': round(time.monotonic()-start, 2), 'device': str(device),
                      'gpu_allocated_mb': round(torch.cuda.memory_allocated()/1024**2, 1) if device.type == 'cuda' else 0,
                      'gpu_peak_mb': round(torch.cuda.max_memory_allocated()/1024**2, 1) if device.type == 'cuda' else 0}
            is_best = accuracy >= best_accuracy
            best_accuracy = max(accuracy, best_accuracy)
            payload = {'format_version': 2, 'task': a.task, 'model_state': cpu_copy(model.state_dict()),
                       'optimizer_state': cpu_copy(optimizer.state_dict()), 'scaler_state': scaler.state_dict(),
                       'rng_state': torch.get_rng_state(), 'meta': details, 'epoch': epoch,
                       'accuracy': accuracy, 'best_accuracy': best_accuracy,
                       'fingerprint': fingerprint, 'batch_size': a.batch_size, 'precision': a.precision}
            if is_best:
                atomic_checkpoint(out / 'best.pt', payload)
            atomic_checkpoint(last_path, payload)
            metrics.write(json.dumps(metric, ensure_ascii=False) + '\n')
            (out / 'run_state.json').write_text(json.dumps({'epoch': epoch, 'target_epochs': a.epochs,
                                                             'device': str(device), 'precision': a.precision}), encoding='utf-8')
            print(json.dumps(metric, ensure_ascii=False), flush=True)
    if should_stop:
        print('Stop requested; last completed epoch checkpoint retained. Use Resume to continue.', flush=True)
        return 130
    print(f'Completed {a.epochs} epochs; best accuracy={best_accuracy:.4f}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
