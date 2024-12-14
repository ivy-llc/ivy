import numbers
import functools
import types
from typing import Callable

import ivy

# Helpers for graph visulaisation #
# ------------------------------- #


def _format_label(cls, shape_or_val, newline=False) -> str:
    ptype_str = (
        "{}".format(cls)
        .replace("'", "")
        .replace(" ", "")
        .replace("<", "")
        .replace(">", "")
        .replace("class", "")
        .split(".")[-1]
    )
    if ivy.exists(shape_or_val):
        if newline:
            return ptype_str + "\n{}".format(shape_or_val)
        return ptype_str + ", {}".format(shape_or_val)
    return ptype_str


def _param_to_label(param):
    return _format_label(param.ptype, param.shape, newline=True)


def _to_label(x):
    return _format_label(
        type(x),
        ivy.default(
            lambda: tuple(x.shape),
            default_val=x if isinstance(x, (str, numbers.Number)) else None,
            catch_exceptions=True,
        ),
    )


def _get_argument_reprs(keys, args, kwargs):
    # fails when an argument can take a variable number
    # of inputs e.g. `*arrays` in `meshgrid`
    try:
        repr = "\n".join(
            [
                "{}: {}".format(k, v)
                for k, v in dict(
                    **dict(
                        [
                            (keys[i] if i < len(keys) else str(i), _to_label(a))
                            for i, a in enumerate(args)
                        ]
                    ),
                    **kwargs,
                ).items()
            ]
        )
    except Exception:
        repr = ""
    return repr


def _get_output_reprs(output):
    return "\n".join(
        [
            "{}: {}".format(k, v)
            for k, v in dict(
                [(str(i), _to_label(a)) for i, a in enumerate(output)]
            ).items()
        ]
    )


def _args_str_from_fn(fn: Callable) -> str:
    return fn.arg_n_kwarg_reprs


def _output_str_from_fn(fn: Callable) -> str:
    if hasattr(fn, "output_reprs"):
        return fn.output_reprs
    else:
        return ""


def _copy_func(f: Callable) -> Callable:
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
