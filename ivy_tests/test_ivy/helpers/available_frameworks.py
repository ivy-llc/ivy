# A list of available backends that can be used for testing.


def available_frameworks():
    available_frameworks_lis = ["numpy", "jax", "tensorflow", "torch", "paddle"]
    try:
        import jax

        assert jax, "jax is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("jax")

    try:
        import tensorflow as tf

        assert tf, "tensorflow is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("tensorflow")

    try:
        import torch

        assert torch, "torch is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("torch")

    try:
        import paddle

        assert paddle, "Paddle is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("paddle")
    return available_frameworks_lis

    try:
        import mxnet

        assert mxnet, "mxnet is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("mxnet")


def ground_truth():
    available_framework_lis = available_frameworks()
    g_truth = ""
    if "tensorflow" in available_framework_lis:
        g_truth = "tensorflow"
    elif "torch" in available_framework_lis:
        g_truth = "torch"
    elif "jax" in available_framework_lis:
        g_truth = "jax"
    elif "paddle" in available_framework_lis:
        g_truth = "paddle"
    elif "mxnet" in available_framework_lis:
        g_truth = "mxnet"
    else:
        g_truth = "numpy"
    return g_truth
