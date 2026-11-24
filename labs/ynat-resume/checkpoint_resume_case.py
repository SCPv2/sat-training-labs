#!/usr/bin/env python3
"""Deliberately stop after a committed S3 checkpoint, then resume in a new process."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from botocore.exceptions import ClientError

from settings import storage_uri
from storage import S3CheckpointStore, upload_results


def run(output, arguments, injection_step=None):
    command = [sys.executable, '/lesson/train.py', '--task', 'text',
               '--bundle', str(arguments.bundle), '--model', str(arguments.model),
               '--output', str(output), '--steps', str(arguments.steps),
               '--train-count', str(arguments.train_count),
               '--eval-count', str(arguments.eval_count),
               '--checkpoint-steps', str(arguments.checkpoint_steps)]
    environment = os.environ.copy()
    environment.pop('SAT_INJECT_CHECKPOINT_FAILURE_STEP', None)
    if injection_step is not None:
        environment['SAT_INJECT_CHECKPOINT_FAILURE_STEP'] = str(injection_step)
    return subprocess.run(command, env=environment, check=False).returncode


def latest_step(store):
    response = store.client.get_object(Bucket=store.bucket, Key=store.key('latest.json'))
    return int(json.loads(response['Body'].read())['step'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path('/bundle'))
    parser.add_argument('--model', type=Path, default=Path('/model'))
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--checkpoint-steps', type=int, default=20)
    parser.add_argument('--train-count', type=int, default=100)
    parser.add_argument('--eval-count', type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.checkpoint_steps < args.steps:
        raise ValueError('Checkpoint step must be between zero and total steps')
    checkpoint_uri = storage_uri(None, 'SAT_CHECKPOINT_URI')
    output_uri = storage_uri(None, 'SAT_OUTPUT_URI')
    if not checkpoint_uri or not output_uri:
        raise ValueError('S3 checkpoint and result URIs are required')
    store = S3CheckpointStore(checkpoint_uri)
    if latest_step_if_exists(store) is not None:
        raise ValueError('Checkpoint prefix must be empty before this controlled exercise')

    first = args.output / 'attempt-one'
    first_code = run(first, args, args.checkpoint_steps)
    marker = first / 'text' / 'intentional-interruption.json'
    if first_code == 0 or not marker.is_file():
        raise RuntimeError('First process did not fail at the injected checkpoint')
    interruption = json.loads(marker.read_text())
    if (interruption.get('step') != args.checkpoint_steps or
            interruption.get('reason') != 'injected-after-s3-commit' or
            latest_step(store) != args.checkpoint_steps):
        raise RuntimeError('S3 checkpoint was not committed before the interruption')
    print(json.dumps({'exercise': 'first_process_failed_after_checkpoint',
                      'step': args.checkpoint_steps, 'exit_code': first_code}), flush=True)

    second = args.output / 'attempt-two'
    code = run(second, args)
    if code:
        raise RuntimeError('Resumed process failed with exit code ' + str(code))
    result_dir = second / 'text'
    comparison = json.loads((result_dir / 'comparison.json').read_text())
    if comparison.get('resumed_from_step') != args.checkpoint_steps:
        raise RuntimeError('Second process did not restore the expected S3 checkpoint')
    if comparison.get('steps') != args.steps or latest_step(store) != args.steps:
        raise RuntimeError('Resumed training did not complete all requested steps')
    validation = {'version': 1, 'intentional_failure': True,
                  'first_exit_code': first_code,
                  'checkpoint_at_failure': args.checkpoint_steps,
                  'resumed_from_step': comparison['resumed_from_step'],
                  'completed_steps': args.steps, 'checkpoint_uri': checkpoint_uri,
                  'first_output': str(first), 'second_output': str(second)}
    (result_dir / 'resume-validation.json').write_text(json.dumps(validation, indent=2))
    upload_results(output_uri, result_dir)
    print(json.dumps({'exercise': 'resumed_and_completed', **validation}), flush=True)


def latest_step_if_exists(store):
    try:
        return latest_step(store)
    except ClientError as error:
        if error.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
            return None
        raise



if __name__ == '__main__':
    main()
