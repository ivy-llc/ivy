# global
import functools
import logging
from typing import Callable

import jax

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend


def _is_jax_frontend_array(x):
    return isinstance(x, jax_frontend.DeviceArray)


def _from_jax_frontend_array_to_ivy_array(x):
    if isinstance(x, jax_frontend.DeviceArray):
        return x.data
    return x


def _from_ivy_array_to_jax_frontend_array(x, nested=False, include_derived=None):
    if nested:
        return ivy.nested_map(x, _from_ivy_array_to_jax_frontend_array, include_derived)
    elif isinstance(x, ivy.Array):
        return jax_frontend.DeviceArray(x)
    return x


def _jax_array_to_ivy_array(x):
    if isinstance(x, jax.numpy.DeviceArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _from_jax_frontend_array_to_ivy_array(_jax_array_to_ivy_array(x))


def _has_nans(x):
    if isinstance(x, ivy.Container):
        return x.has_nans()
    elif isinstance(x, ivy.Array) or ivy.is_native_array(x):
        return ivy.isnan(x).any()
    elif isinstance(x, tuple):
        return any(_has_nans(xi) for xi in x)
    else:
        return False


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        # check if kwargs contains an out argument, and if so, remove it
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True
        # convert all arrays in the inputs to ivy.Array instances
        new_args = ivy.nested_map(args, _to_ivy_array, include_derived={tuple: True})
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}
        )
        # add the original out argument back to the keyword arguments
        if has_out:
            new_kwargs["out"] = out
        return fn(*new_args, **new_kwargs)

    return new_fn


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        # call unmodified function
        ret = fn(*args, **kwargs)
        # convert all arrays in the return to `jax_frontend.DeviceArray` instances
        return _from_ivy_array_to_jax_frontend_array(
            ret, nested=True, include_derived={tuple: True}
        )

    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def handle_nans(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Checks for the existence of nans in all arrays in the `args`
        and `kwargs`. The presence of nans is then handled depending
        on the enabled `nan_policy`.

        Following policies apply:
        raise_exception: raises an exception in case nans are present
        warns: warns a user in case nans are present
        nothing: does nothing

        Parameters
        ----------
        args
            The arguments to be passed to the function.
        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with handling of inputs based
            on the selected `nan_policy`.
        """
        # skip the check if the current nan policy is `nothing``
        if ivy.get_nan_policy() == "nothing":
            return fn(*args, **kwargs)

        # check all args and kwards for presence of nans
        result = ivy.nested_any(args, _has_nans) or ivy.nested_any(
            kwargs, _has_nans)

        if result:
            # handle nans based on the selected policy
            if ivy.get_nan_policy() == "raise_exception":
                raise ivy.exceptions.IvyException(
                    "Nans are not allowed in `raise_exception` policy.")
            elif ivy.get_nan_policy() == "warns":
                logging.warning("Nans are present in the input.")
        
        return fn(*args, **kwargs)

    new_fn.handle_nans = True
    return new_fn
