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
    frameworks = available_frameworks()
    gt_frameworks = ["tensorflow", "torch", "jax", "paddle", "mxnet"]
    return next((gt_framework
                 for gt_framework in gt_frameworks if
                 gt_framework in frameworks),
                "numpy")
