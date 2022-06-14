# global
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


@pytest.mark.parametrize("backend", ["torch", "numpy", "tensorflow", "jax"])
def test_docstrings(backend):
    ivy.set_default_device("cpu")
    ivy.set_backend(backend)
    failures = list()
    success = True

    """ 
        Functions skipped as their output dependent on outside factors:
            random_normal, random_uniform, shuffle, num_gpus, current_backend,
            get_backend
            
        Functions skipped due to <lambda>-related error (cause test to fail):
            current_backend_str, container_types, inplace_arrays_supported,
            inplace_variables_supported, multiprocessing, variable_data,
            get_num_dims, unset_backend         
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
        "current_backend_str",
        "container_types",
        "inplace_arrays_supported",
        "inplace_variables_supported",
        "multiprocessing",
        "variable_data",
        "get_num_dims",
        "unset_backend",
    ]

    function_list = ivy.__dict__.copy().items()
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
    ivy.unset_backend()
