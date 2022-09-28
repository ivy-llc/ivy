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


def handle_numpy_casting(fn: Callable) -> Callable:
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
        # TODO: dynamic arg number
        ivy.assertions.check_elem_in_list(
            casting,
            ["no", "equiv", "safe", "same_kind", "safe"],
            message="casting must be one of [no, equiv, safe, same_kind, safe]",
        )
        if casting == "no" or casting == "equiv":
            ivy.assertions.check_equal(
                ivy.dtype(args[0]),
                ivy.dtype(args[1]),
                message="casting is 'no', input types must be the same",
            )
            if ivy.exists(dtype):
                ivy.assertions.check_equal(
                    ivy.as_ivy_dtype(dtype),
                    ivy.dtype(args[0]),
                    message="casting is 'no', dtype must be the same as input types",
                )
        elif casting == "safe":
            # TODO: test if this is required or handled in backend
            # promoted_type = ivy.promote_types(ivy.dtype(args[0]), ivy.dtype(args[1]))
            if ivy.exists(dtype):
                ivy.assertions.check_all_or_any_fn(
                    args[0],
                    args[1],
                    fn=lambda x: ivy.can_cast(x, dtype),
                    type="all",
                    message="type of input is incompatible with dtype: {}".format(
                        dtype
                    ),
                )
                args[0] = ivy.astype(args[0], dtype)
                args[1] = ivy.astype(args[1], dtype)
        elif casting == "same_kind":
            if ivy.exists(dtype):
                ivy.assertions.check_all_or_any_fn(
                    args[0],
                    args[1],
                    fn=lambda x: _is_same_kind_or_safe(
                        ivy.dtype(x), ivy.as_ivy_dtype(dtype)
                    ),
                    type="all",
                    message="type of input is incompatible with dtype: {}".format(
                        dtype
                    ),
                )
                args[0] = ivy.astype(args[0], dtype)
                args[1] = ivy.astype(args[1], dtype)
        else:
            args[0] = ivy.astype(args[0], dtype)
            args[1] = ivy.astype(args[1], dtype)
        return fn(*args, **kwargs)

    new_fn.handle_numpy_casting = True
    return new_fn
