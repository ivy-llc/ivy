# A list of available backends that can be used for testing.

available_frameworks = ["numpy", "jax", "tensorflow", "torch"]

try:
    import jax

    assert jax, "jax is imported to see if the user has it installed"
except ImportError:
    available_frameworks.remove("jax")

try:
    import tensorflow as tf

    assert tf, "tensorflow is imported to see if the user has it installed"
except ImportError:
    available_frameworks.remove("tensorflow")

try:
    import torch

    assert torch, "torch is imported to see if the user has it installed"
except ImportError:
    available_frameworks.remove("torch")

if "tensorflow" in available_frameworks:
    ground_truth = "tensorflow"
elif "torch" in available_frameworks:
    ground_truth = "torch"
elif "jax" in available_frameworks:
    ground_truth = "jax"
else:
    ground_truth = "numpy"
