import ivy
import ivy.functional.frontends.builtins as builtins_frontend
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def abs(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def range(start, /, stop=None, step=1):
    if not stop:
        return ivy.arange(0, stop=start, step=step)
    return ivy.arange(start, stop=stop, step=step)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def all(iterable):
    return ivy.all(iterable)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def any(iterable):
    return ivy.any(iterable)


@from_zero_dim_arrays_to_scalar
def round(number, ndigits=None):
    if not ndigits:
        return ivy.round(number)
    return ivy.round(number, decimals=ndigits)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def min(*args, default=None, key=None):
    # arguments are empty
    if len(args) == 0:
        # in this case default should be provided
        if not default:
            raise ValueError("default must be provided for empty input")
        return default

    # this means we deal with iterable rather than separate arguments
    elif len(args) == 1 and not ivy.isscalar(args[0]):
        # pass iterable to the same func
        return builtins_frontend.min(*args[0], default=default, key=key)

    # if keyfunc provided, map all args to it
    if key:
        mapped_args = ivy.map(key, constant=None, unique={"x": args}, mean=False)
        idx = ivy.argmin(mapped_args)
        # argmin always returns array, convert it to scalar
        idx = ivy.to_scalar(idx)
        return args[idx]

    return ivy.min(args)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def max(*args, default=None, key=None):
    # arguments are empty
    if len(args) == 0:
        # in this case default should be provided
        if not default:
            raise ValueError("default must be provided for empty input")
        return default

    # this means we deal with iterable rather than separate arguments
    elif len(args) == 1 and not ivy.isscalar(args[0]):
        # pass iterable to the same func
        return builtins_frontend.max(*args[0], default=default, key=key)

    # if keyfunc provided, map all args to it
    if key:
        mapped_args = ivy.map(key, constant=None, unique={"x": args}, mean=False)
        idx = ivy.argmax(mapped_args)
        # argmin always returns array, convert it to scalar
        idx = ivy.to_scalar(idx)
        return args[idx]

    return ivy.max(args)
