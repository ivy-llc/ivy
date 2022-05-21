# global
import ivy
import pdb
import logging

# local
from ivy.func_wrapper import _wrap_or_unwrap_functions, NON_WRAPPED_FUNCTIONS


queue_timeout = None
debug_mode_val = False


# Methods #


def _wrap_method_for_debugging(fn):

    if hasattr(fn, "__name__") and (
        fn.__name__[0] == "_"
        or fn.__name__
        in set(
        NON_WRAPPED_FUNCTIONS
        + ["has_nans", "is_array", "value_is_nan", "reduce_sum", "to_scalar"]
        )
    ):
        return fn

    if hasattr(fn, "wrapped_for_debugging") and fn.wrapped_for_debugging:
        return fn

    def _method_wrapped(*args, **kwargs):
        def _check_nans(x):
            if ivy.is_native_array(x) and ivy.has_nans(x):
                if debug_mode_val == "exception":
                    raise Exception("found nans in {}".format(x))
                else:
                    logging.error("found nans in {}".format(x))
                    pdb.set_trace()
            return x

        ivy.nested_map(args, _check_nans)
        ivy.nested_map(kwargs, _check_nans)
        ret = fn(*args, **kwargs)
        ivy.nested_map(ret, _check_nans)
        return ret

    if hasattr(fn, "__name__"):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped_for_debugging = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method_from_debugging(method_wrapped):

    if (
        not hasattr(method_wrapped, "wrapped_for_debugging")
        or not method_wrapped.wrapped_for_debugging
    ):
        return method_wrapped
    return method_wrapped.inner_fn


def _wrap_methods_for_debugging():
    return _wrap_or_unwrap_functions(_wrap_method_for_debugging)


def _unwrap_methods_from_debugging():
    return _wrap_or_unwrap_functions(_unwrap_method_from_debugging)


# Mode #


def set_debug_mode(debug_mode_in="exception"):
    assert debug_mode_in in ["breakpoint", "exception"]
    global debug_mode_val
    debug_mode_val = debug_mode_in
    global queue_timeout
    queue_timeout = ivy.queue_timeout()
    ivy.set_queue_timeout(None)
    _wrap_methods_for_debugging()


def set_breakpoint_debug_mode():
    set_debug_mode("breakpoint")


def set_exception_debug_mode():
    set_debug_mode("exception")


def unset_debug_mode():
    global debug_mode_val
    debug_mode_val = False
    _unwrap_methods_from_debugging()
    global queue_timeout
    ivy.set_queue_timeout(queue_timeout)


def debug_mode():
    return debug_mode_val
