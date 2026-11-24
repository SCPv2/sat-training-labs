"""Hugging Face Trainer checkpoint upload and resume helpers."""
import hashlib
import json
import os
from pathlib import Path

from transformers import TrainerCallback

from storage import S3CheckpointStore, latest_local_checkpoint


def training_fingerprint(**values):
    encoded = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class S3CheckpointCallback(TrainerCallback):
    def __init__(self, store, fingerprint):
        self.store = store
        self.fingerprint = fingerprint

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint = Path(args.output_dir) / ('checkpoint-' + str(state.global_step))
        manifest = self.store.publish(checkpoint, self.fingerprint)
        print(json.dumps({'checkpoint_uploaded': manifest['checkpoint'],
                          'step': manifest['step']}), flush=True)
        inject_step = os.environ.get('SAT_INJECT_CHECKPOINT_FAILURE_STEP')
        if inject_step and inject_step.isdecimal() and int(inject_step) == state.global_step:
            marker = Path(args.output_dir).parent / 'intentional-interruption.json'
            marker.write_text(json.dumps({'checkpoint': manifest['checkpoint'],
                                          'step': state.global_step, 'reason': 'injected-after-s3-commit'}))
            print('SAT_EXPECTED_INTERRUPT checkpoint-' + str(state.global_step), flush=True)
            raise RuntimeError('Intentional checkpoint exercise interruption at step ' + str(state.global_step))
        return control


def prepare_resume(checkpoint_root, checkpoint_uri, fingerprint):
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / '.training-fingerprint'
    if marker.exists() and marker.read_text() != fingerprint:
        raise ValueError('Local checkpoint directory belongs to another training recipe')
    if checkpoint_uri:
        store = S3CheckpointStore(checkpoint_uri)
        checkpoint = store.restore_latest(root, fingerprint)
        marker.write_text(fingerprint)
        return checkpoint, store
    checkpoint = latest_local_checkpoint(root)
    if checkpoint and not marker.exists():
        raise ValueError('Local checkpoint has no training fingerprint')
    marker.write_text(fingerprint)
    return checkpoint, None
