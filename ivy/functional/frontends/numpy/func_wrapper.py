# global
import functools
from typing import Callable, Any
import inspect
import platform

# local
import ivy
from ivy.functional.frontends.numpy.ndarray.ndarray import ndarray
import ivy.functional.frontends.numpy as np_frontend


def handle_numpy_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, dtype=None, **kwargs):
        if len(args) > (dtype_pos + 1):
            dtype = args[dtype_pos]
            kwargs = {
                **dict(
                    zip(
                        list(inspect.signature(fn).parameters.keys())[
                            dtype_pos + 1 : len(args)
                        ],
                        args[dtype_pos + 1 :],
                    )
                ),
                **kwargs,
            }
            args = args[:dtype_pos]
        elif len(args) == (dtype_pos + 1):
            dtype = args[dtype_pos]
            args = args[:-1]
        return fn(*args, dtype=np_frontend.to_ivy_dtype(dtype), **kwargs)

    dtype_pos = list(inspect.signature(fn).parameters).index("dtype")
    new_fn.handle_numpy_dtype = True
    return new_fn


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


def _assert_no_scalar(args, dtype):
    if args:
        first_arg = args[0]
        ivy.assertions.check_all_or_any_fn(
            *args,
            fn=lambda x: type(x) == type(first_arg),
            type="all",
            message="type of input is incompatible with dtype {}".format(dtype),
        )
        if ivy.exists(dtype):
            if ivy.is_int_dtype(dtype):
                check_dtype = int
            elif ivy.is_float_dtype(dtype):
                check_dtype = float
            elif ivy.is_bool_dtype(dtype):
                check_dtype = bool
        ivy.assertions.check_equal(
            type(args[0]),
            check_dtype,
            message="type of input is incompatible with dtype {}".format(dtype),
        )


def _assert_no_array(args, dtype):
    if args:
        first_arg = args[0]
        fn_func = ivy.as_ivy_dtype(dtype) if ivy.exists(dtype) else ivy.dtype(first_arg)
        ivy.assertions.check_all_or_any_fn(
            *args,
            fn=lambda x: ivy.dtype(x) == fn_func,
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

        # check if scalar exists and convert them to 0-dim arrays,
        # so that their dtypes are handled correctly
        args_scalar_idxs = ivy.nested_argwhere(
            args, lambda x: isinstance(x, (int, float, bool))
        )
        args_scalar_to_check = ivy.multi_index_nest(args, args_scalar_idxs)
        args_idxs = ivy.nested_argwhere(args, ivy.is_array)
        args_to_check = ivy.multi_index_nest(args, args_idxs)
        kwargs_idxs = ivy.nested_argwhere(kwargs, ivy.is_array)
        kwargs_idxs.remove(["out"]) if ["out"] in kwargs_idxs else kwargs_idxs
        kwargs_to_check = ivy.multi_index_nest(kwargs, kwargs_idxs)

        if casting in ["no", "equiv"]:
            _assert_no_scalar(args_scalar_to_check, dtype)
            _assert_no_array(args_to_check, dtype)
            # Todo: cross type check, kwargs check
        elif ivy.exists(dtype):
            assert_fn = None
            if casting == "safe":
                assert_fn = lambda x: np_frontend.can_cast(x, ivy.as_ivy_dtype(dtype))
            elif casting == "same_kind":
                assert_fn = lambda x: np_frontend.can_cast(
                    x, ivy.as_ivy_dtype(dtype), casting="same_kind"
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


def handle_numpy_casting_special(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, casting="same_kind", dtype=None, **kwargs):
        """
        Check numpy casting type for special cases where output must be type bool.

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
        if ivy.exists(dtype):
            ivy.assertions.check_equal(
                ivy.as_ivy_dtype(dtype),
                "bool",
                message="output is compatible with bool only",
            )

        return fn(*args, **kwargs)

    new_fn.handle_numpy_casting_special = True
    return new_fn


def _numpy_frontend_to_ivy(x: Any) -> Any:
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    else:
        return x


def _ivy_to_numpy(x: Any) -> Any:
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = ndarray(0)  # TODO Find better initialisation workaround
        a.ivy_array = x
        a.dtype = ivy.dtype(x)
        return a
    else:
        return x


def _ivy_to_numpy_order_F(x: Any) -> Any:
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = ndarray(0, order="F")  # TODO Find better initialisation workaround
        a.ivy_array = x
        a.dtype = ivy.dtype(x)
        return a
    else:
        return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _numpy_frontend_to_ivy(_native_to_ivy_array(x))


def _check_C_order(x):
    if isinstance(x, ivy.Array):
        return True
    elif isinstance(x, ndarray):
        if x._f_contiguous:
            return False
        else:
            return True
    else:
        return None


def _set_order(args, order):
    ivy.assertions.check_elem_in_list(
        order,
        ["C", "F", "A", "K", None],
        message="order must be one of 'C', 'F', 'A', or 'K'",
    )
    if order in ["K", "A", None]:
        check_order = ivy.nested_map(
            args, _check_C_order, include_derived={tuple: True}, shallow=False
        )
        if all(v is None for v in check_order) or any(
            ivy.multi_index_nest(check_order, ivy.all_nested_indices(check_order))
        ):
            order = "C"
        else:
            order = "F"
    return order


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
        ivy_args = ivy.nested_map(args, _to_ivy_array, include_derived={tuple: True})
        ivy_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    new_fn.inputs_to_ivy_arrays = True
    return new_fn


def outputs_to_numpy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, order="K", **kwargs):
        """
        Calls the function, and then converts all `ivy.Array` instances returned
        by the function into `ndarray` instances.
           The return of the function, with ivy arrays as numpy arrays.
        """
        # handle order and call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        ivy.set_default_int_dtype(
            "int64"
        ) if platform.system() != "Windows" else ivy.set_default_int_dtype("int32")
        ivy.set_default_float_dtype("float64")
        if contains_order:
            if len(args) >= (order_pos + 1):
                order = args[order_pos]
                args = args[:-1]
            order = _set_order(args, order)
            try:
                ret = fn(*args, order=order, **kwargs)
            finally:
                ivy.unset_default_int_dtype()
                ivy.unset_default_float_dtype()
        else:
            try:
                ret = fn(*args, **kwargs)
            finally:
                ivy.unset_default_int_dtype()
                ivy.unset_default_float_dtype()
        if not ivy.get_array_mode():
            return ret
        # convert all returned arrays to `ndarray` instances
        if order == "F":
            return ivy.nested_map(
                ret, _ivy_to_numpy_order_F, include_derived={tuple: True}
            )
        else:
            return ivy.nested_map(ret, _ivy_to_numpy, include_derived={tuple: True})

    if "order" in list(inspect.signature(fn).parameters.keys()):
        contains_order = True
        order_pos = list(inspect.signature(fn).parameters).index("order")
    else:
        contains_order = False
    new_fn.outputs_to_numpy_arrays = True
    return new_fn


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wraps `fn` so that input arrays are all converted to `ivy.Array` instances
    and return arrays are all converted to `ndarray` instances.
    """
    return outputs_to_numpy_arrays(inputs_to_ivy_arrays(fn))


def from_zero_dim_arrays_to_scalar(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Calls the function, and then converts all 0 dimensional array instances in
        the function to float numbers if out argument is not provided.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with 0 dimensional arrays as float numbers.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        if ("out" in kwargs and kwargs["out"] is None) or "out" not in kwargs:
            if isinstance(ret, tuple):
                # converting every scalar element of the tuple to float
                data = tuple([ivy.native_array(i) for i in ret])
                data = ivy.copy_nest(data, to_mutable=True)
                ret_idx = ivy.nested_argwhere(data, lambda x: x.shape == ())
                try:
                    ivy.map_nest_at_indices(
                        data,
                        ret_idx,
                        lambda x: np_frontend.numpy_dtype_to_scalar[ivy.dtype(x)](x),
                    )
                except KeyError:
                    raise ivy.exceptions.IvyException(
                        "Casting to specified type is unsupported"
                    )
                return tuple(data)
            else:
                # converting the scalar to float
                data = ivy.native_array(ret)
                if data.shape == ():
                    try:
                        return np_frontend.numpy_dtype_to_scalar[ivy.dtype(data)](data)
                    except KeyError:
                        raise ivy.exceptions.IvyException(
                            f"Casting to {ivy.dtype(data)} is unsupported"
                        )
        return ret

    new_fn.from_zero_dim_arrays_to_scalar = True
    return new_fn


def handle_numpy_out(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, out=None, **kwargs):
        if len(args) > (out_pos + 1):
            out = args[out_pos]
            kwargs = {
                **dict(
                    zip(
                        list(inspect.signature(fn).parameters.keys())[
                            out_pos + 1 : len(args)
                        ],
                        args[out_pos + 1 :],
                    )
                ),
                **kwargs,
            }
            args = args[:out_pos]
        elif len(args) == (out_pos + 1):
            out = args[out_pos]
            args = args[:-1]
        if hasattr(out, "ivy_array"):
            return fn(*args, out=out.ivy_array, **kwargs)
        return fn(*args, out=out, **kwargs)

    out_pos = list(inspect.signature(fn).parameters).index("out")
    new_fn.handle_numpy_out = True
    return new_fn
