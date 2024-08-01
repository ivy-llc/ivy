import functools
import ivy
import ivy.functional.backends.keras as keras_backend
import ivy.functional.backends.jax as jax_backend
import ivy.functional.backends.tensorflow as tf_backend
import ivy.functional.backends.torch as torch_backend
import os
from typing import Callable


def use_keras_backend_framework(fn: Callable) -> Callable:
    """
    Wraps the function such that it instead calls the equivalent function
    from the ivy backend equivalent to the keras backend currently set.
    """

    @functools.wraps(fn)
    def _use_keras_backend_framework(*args, **kwargs):
        keras_backend = os.getenv("KERAS_BACKEND", default="tensorflow").lower()
        assert keras_backend in ["jax", "tensorflow", "torch"]

        if keras_backend == "jax":
            ivy_keras_backend = jax_backend
        elif keras_backend == "tensorflow":
            ivy_keras_backend = tf_backend
        elif keras_backend == "torch":
            ivy_keras_backend = torch_backend
        else:
            # default to tensorflow backend
            # TODO: raise warning?
            ivy_keras_backend = tf_backend

        backend_fn = getattr(ivy_keras_backend, fn.__name__)
        return backend_fn(*args, **kwargs)

    return _use_keras_backend_framework
