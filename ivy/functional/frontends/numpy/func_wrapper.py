import ivy
import functools
from typing import Callable


def _is_same_kind_or_safe(t1, t2):
    if ivy.is_float_dtype(t1):
        return ivy.is_float_dtype(t2) or ivy.can_cast(t1, t2)
    elif ivy.is_uint_dtype(t1):
        return ivy.is_uint_dtype(t2) or ivy.can_cast(t1, t2)
    elif ivy.is_int_dtype(t1):
        return ivy.is_int_dtype(t2) or ivy.can_cast(t1, t2)
    elif ivy.is_bool_dtype(t1):
        return ivy.is_bool_dtype(t2) or ivy.can_cast(t1, t2)
    raise ivy.exceptions.IvyException(
        "dtypes of input must be float, int, uint, or bool"
    )


def _process_args_and_dtype(args, dtype, num_args):
    return [
        ivy.astype(args[i], ivy.as_ivy_dtype(dtype)) if i < num_args else args[i]
        for i in range(len(args))
    ]


def _assert_args_and_fn(args, dtype, fn):
    ivy.assertions.check_all_or_any_fn(
        *args,
        fn=fn,
        type="all",
        message="type of input is incompatible with dtype: {}".format(dtype),
    )


def handle_numpy_casting(num_args: int = 1) -> Callable:
    def inner_decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def new_fn(*args, casting="same_kind", dtype=None, **kwargs):
            """
            Check numpy casting type.

            Parameters
            ----------
            args
                The arguments to be passed to the function.

            kwargs
                The keyword arguments to be passed to the function.

            Returns
            -------
                The return of the function, or raise IvyException if error is thrown.
            """
            ivy.assertions.check_elem_in_list(
                casting,
                ["no", "equiv", "safe", "same_kind", "unsafe"],
                message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
            )
            args = list(args)
            if casting == "no" or casting == "equiv":
                _assert_args_and_fn(
                    args[:num_args],
                    dtype,
                    fn=lambda x: ivy.dtype(x) == ivy.dtype(args[0]),
                )
                if ivy.exists(dtype):
                    ivy.assertions.check_equal(
                        ivy.as_ivy_dtype(dtype),
                        ivy.dtype(args[0]),
                        message="casting is {}, dtype must match input types".format(
                            casting
                        ),
                    )
            elif casting == "safe":
                if ivy.exists(dtype):
                    _assert_args_and_fn(
                        args[:num_args],
                        dtype,
                        fn=lambda x: ivy.can_cast(x, ivy.as_ivy_dtype(dtype)),
                    )
                    args = _process_args_and_dtype(args, dtype, num_args)
            elif casting == "same_kind":
                if ivy.exists(dtype):
                    _assert_args_and_fn(
                        args[:num_args],
                        dtype,
                        fn=lambda x: _is_same_kind_or_safe(
                            ivy.dtype(x), ivy.as_ivy_dtype(dtype)
                        ),
                    )
                    args = _process_args_and_dtype(args, dtype, num_args)
            else:
                args = _process_args_and_dtype(args, dtype, num_args)
            return fn(*args, **kwargs)

        new_fn.handle_numpy_casting = True
        return new_fn

    return inner_decorator
