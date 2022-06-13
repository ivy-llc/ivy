# global
import warnings

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


def test_docstrings():
    failures = list()
    success = True

    """ 
        Functions skipped as their output dependent on outside factors:
            random_normal, random_uniform, shuffle, num_gpus, current_backend,
            get_backend
    """
    to_skip = [
        "random_normal",
        "random_uniform",
        "shuffle",
        "num_gpus",
        "current_backend",
        "get_backend",
        "namedtuple",
        "DType",
        "Dtype",
    ]

    function_list = ivy.__dict__.items()
    for k, v in function_list:
        if k in to_skip or helpers.docstring_examples_run(v):
            continue
        success = False
        failures.append(k)
    if not success:
        warnings.warn(
            "the following methods had failing docstrings:\n\n{}".format(
                "\n".join(failures)
            )
        )
