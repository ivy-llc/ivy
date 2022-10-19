# global
from typing import Callable
import functools

# local
import ivy
import ivy.functional.frontends.tensorflow as frontend


def tensorflow_array_to_ivy(x):
    if isinstance(x, frontend.Tensor):
        return x.data
    return x


def ivy_array_to_tensorflow(x):
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        return frontend.Tensor(x.data)
    return x


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Converts all `TensorFlow.Tensor` instances in both the positional and keyword
        arguments into `ivy.Array` instances, and then calls the function with the
        updated arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays passed in the arguments.
        """
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True

        # convert all arrays in the inputs to ivy.Array instances
        ivy_args = ivy.nested_map(args, tensorflow_array_to_ivy, include_derived=True)
        ivy_kwargs = ivy.nested_map(
            kwargs, tensorflow_array_to_ivy, include_derived=True
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    new_fn.inputs_to_ivy_arrays = True
    return new_fn


def outputs_to_tensorflow_array(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Calls the function, and then converts all `tensorflow.Tensor` instances in
        the function return into `ivy.Array` instances.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays as tensorflow.Tensor arrays.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.Tensorflow.tensor` instances
        return ivy.nested_map(
            ret, ivy_array_to_tensorflow, include_derived={tuple: True}
        )

    new_fn.outputs_to_tensorflow_array = True
    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    return outputs_to_tensorflow_array(inputs_to_ivy_arrays(fn))
