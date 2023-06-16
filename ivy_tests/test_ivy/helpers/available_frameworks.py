# A list of available backends that can be used for testing.


def available_frameworks():
    frameworks = ["numpy", "jax", "tensorflow", "torch", "paddle"]
    for fw in frameworks:
        try:
            exec(f"import {fw}")
            assert exec(fw), f"{fw} is imported to see if the user has it installed"
        except ImportError:
            frameworks.remove(fw)

    return frameworks


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
