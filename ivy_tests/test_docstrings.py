# global
import warnings
import torch

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
        "shuffle",
        "num_gpus",
        "current_backend",
        "get_backend",
        "namedtuple",
        "DType",
        "Dtype",
        "multinomial",
        "num_cpu_cores",
        "get_all_ivy_arrays_on_dev",
        "num_ivy_arrays_on_dev",
    ]
    # for k, v in ivy.__dict__.copy().items():
    #     if k in to_skip or helpers.docstring_examples_run(v):
    #         continue
    #     success = False
    #     failures.append(k)
    # if not success:
    #     warnings.warn(
    #         "the following methods had failing docstrings:\n\n{}".format(
    #             "\n".join(failures)
    #         )
    #     )
    x = ivy.native_array([[1., 2.], [4., 5.],[7., 8.]])
    y = ivy.array([[1.],  [0.], [0.]])
    z = ivy.cross(x, y, axis=0)
    print(x.shape[-1], x.shape[-1])
    print(z)


    ivy.unset_backend()
