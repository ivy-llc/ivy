# global
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


@pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch"])
def test_docstrings(backend):
    ivy.set_default_device("cpu")
    ivy.set_backend(backend)
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
        "randint",
        "shuffle",
        "num_gpus",
        "current_backend",
        "get_backend",
        "namedtuple",
        "invalid_dtype",
        "DType",
        "Dtype",
        "multinomial",
        "num_cpu_cores",
        "get_all_ivy_arrays_on_dev",
        "num_ivy_arrays_on_dev",
        "total_mem_on_dev",
        "used_mem_on_dev",
        "percent_used_mem_on_dev",
        "function_unsupported_dtypes",
        "randint",
        "unique_counts",
        "unique_all",
        "total_mem_on_dev",
    ]
    # the temp skip list consists of function which have an issue with their
    # implementation
    skip_list_temp = [
        "outer",
        "argmax",
        "split",
        "det",
        "cumprod",
        "where",
        "conv3d_transpose",
    ]

    # skip list for array and container docstrings
    skip_arr_cont = [
        "cross_entropy",
    ]
    currently_being_worked_on = []

    # comment out the line below in future to check for the functions in temp skip list
    to_skip += skip_list_temp + currently_being_worked_on

    for k, v in ivy.__dict__.copy().items():
        if k == "Array":
            for method_name in dir(v):
                method = getattr(ivy.Array, method_name)
                if method_name in skip_arr_cont \
                        or helpers.gradient_incompatible_function(
                        fn=method) \
                        or helpers.docstring_examples_run(
                        fn=method, from_array=True):
                    continue
                success = False
                failures.append("Array." + method_name)

        elif k == "Container":
            for method_name in dir(v):
                method = getattr(ivy.Container, method_name)
                if method_name in skip_arr_cont \
                        or helpers.gradient_incompatible_function(
                        fn=method) \
                        or helpers.docstring_examples_run(
                        fn=method, from_container=True):
                    continue
                success = False
                failures.append("Container." + method_name)

        else:
            if k in to_skip \
                    or helpers.gradient_incompatible_function(
                    fn=v) \
                    or helpers.docstring_examples_run(
                    fn=v):
                continue
            success = False
            failures.append(k)

    if not success:
        assert (
            success
        ), "\nThe following methods had failing docstrings:\n\n{}\n".format(
            "\n".join(failures)
        )

    ivy.unset_backend()
