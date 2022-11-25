# A list of available backends that can be used for testing.

available_frameworks = ["numpy", "jax", "tensorflow", "torch"]

try:
    import jax
except ImportError:
    available_frameworks.remove("jax")
try:
    import tensorflow as tf
except ImportError:
    available_frameworks.remove("tensorflow")
try:
    import torch
except ImportError:
    available_frameworks.remove("torch")