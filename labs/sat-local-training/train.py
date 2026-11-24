"""Editable entry point: training logic lives in training/train.py, architectures in model.py.

Run `python train.py --help` for options. Change training/train.py to customize the loop.
"""
from training.train import main

if __name__ == '__main__':
    raise SystemExit(main())
