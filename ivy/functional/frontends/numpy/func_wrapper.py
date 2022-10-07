# global
import functools
from typing import Callable, Any

# local
import ivy
from ndarray import ndarray


def _numpy_to_ivy(x: Any) -> Any:
    if isinstance(x, ndarray):
        return x.data
    else:
        return x


def _ivy_to_numpy(x: Any) -> Any:
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        return ndarray(x)
    else:
        return x


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Converts all `ndarray` instances in both the positional and keyword
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
        if "out" in kwargs:
            out = kwargs["out"]
            has_out = True
        # convert all arrays in the inputs to ivy.Array instances
        ivy_args = ivy.nested_map(args, _numpy_to_ivy, include_derived={tuple: True})
        ivy_kwargs = ivy.nested_map(kwargs, _numpy_to_ivy, include_derived={tuple: True})
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)
    new_fn.inputs_to_ivy_arrays = True
    return new_fn


def outputs_to_numpy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Calls the function, and then converts all `ivy.Array` instances returned
        by the function into `ndarray` instances.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays as numpy arrays.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)
        if not ivy.get_array_mode():
            return ret
        # convert all returned arrays to `ndarray` instances
        return ivy.nested_map(ret, _ivy_to_numpy, include_derived={tuple: True})
    new_fn.outputs_to_numpy_arrays = True
    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wraps `fn` so that input arrays are all converted to `ivy.Array` instances
    and return arrays are all converted to `ndarray` instances.
    """
    return outputs_to_numpy_arrays(inputs_to_ivy_arrays(fn))
