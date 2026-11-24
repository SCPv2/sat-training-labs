"""Small CNN / MLP model definitions used by the local training engine.

Both models work without a network connection and on CPU or low-VRAM CUDA devices.
"""
from training.common import make_model, make_images

__all__ = ['make_model', 'make_images']
