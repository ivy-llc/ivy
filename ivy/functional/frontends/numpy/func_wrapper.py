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


def _assert_args_and_fn(args, kwargs, dtype, fn):
    ivy.assertions.check_all_or_any_fn(
        *args,
        fn=fn,
        type="all",
        message="type of input is incompatible with dtype: {}".format(dtype),
    )
    ivy.assertions.check_all_or_any_fn(
        *kwargs,
        fn=fn,
        type="all",
        message="type of input is incompatible with dtype: {}".format(dtype),
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
        ivy.assertions.check_elem_in_list(
            casting,
            ["no", "equiv", "safe", "same_kind", "unsafe"],
            message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
        )
        args = list(args)
        args_idxs = ivy.nested_argwhere(args, ivy.is_array)
        args_to_check = ivy.multi_index_nest(args, args_idxs)
        kwargs_idxs = ivy.nested_argwhere(kwargs, ivy.is_array)
        kwargs_to_check = ivy.multi_index_nest(kwargs, kwargs_idxs)
        if casting == "no" or casting == "equiv":
            # to check
            _assert_args_and_fn(
                args_to_check,
                kwargs_to_check,
                dtype,
                fn=lambda x: ivy.dtype(x) == ivy.dtype(args[0]),
            )
            # if ivy.exists(dtype):
            #     # if check_output_dtype:
            #
            #     ivy.assertions.check_equal(
            #         ivy.as_ivy_dtype(dtype),
            #         ivy.dtype(args[0]),
            #         message="casting is {}, dtype must match input types".format(
            #             casting
            #         ),
            #     )
        elif ivy.exists(dtype):
            assert_fn = None
            if casting == "safe":
                assert_fn = lambda x: ivy.can_cast(x, ivy.as_ivy_dtype(dtype))
            elif casting == "same_kind":
                assert_fn = lambda x: _is_same_kind_or_safe(
                    ivy.dtype(x), ivy.as_ivy_dtype(dtype)
                )
            if ivy.exists(assert_fn):
                _assert_args_and_fn(
                    args_to_check,
                    kwargs_to_check,
                    dtype,
                    fn=assert_fn,
                )
            ivy.map_nest_at_indices(
                args, args_idxs, lambda x: ivy.astype(x, ivy.as_ivy_dtype(dtype))
            )
            ivy.map_nest_at_indices(
                kwargs, kwargs_idxs, lambda x: ivy.astype(x, ivy.as_ivy_dtype(dtype))
            )

        return fn(*args, **kwargs)

    new_fn.handle_numpy_casting = True
    return new_fn
