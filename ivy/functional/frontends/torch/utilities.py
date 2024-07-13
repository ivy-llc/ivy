import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes
import inspect


# --- Helpers --- #
# --------------- #


@to_ivy_arrays_and_back
def _assert(condition, message):
    if not condition:
        raise Exception(message)
    else:
        return True


# --- Main --- #
# ------------ #


@with_supported_dtypes({"2.2 and above": ("int64",)}, "torch")
@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)


def if_else(cond_fn, body_fn, orelse_fn, vars):
    cond_keys = inspect.getfullargspec(cond_fn).args
    cond_vars = dict(zip(cond_keys, vars))
    return ivy.if_else(cond_fn, body_fn, orelse_fn, cond_vars)


@to_ivy_arrays_and_back
def result_type(tensor, other):
    return ivy.result_type(tensor, other)
