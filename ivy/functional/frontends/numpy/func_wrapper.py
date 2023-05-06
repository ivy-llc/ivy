# global
import functools
from typing import Callable, Any
import inspect
import platform

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


# Helpers #
# ------- #


# general casting
def _assert_array(args, dtype, scalar_check=False, casting="safe"):
    if args and dtype:
        if not scalar_check:
            ivy.utils.assertions.check_all_or_any_fn(
                *args,
                fn=lambda x: np_frontend.can_cast(
                    x, ivy.as_ivy_dtype(dtype), casting=casting
                ),
                type="all",
                message="type of input is incompatible with dtype: {}".format(dtype),
            )
        else:
            assert_fn = None if casting == "safe" else ivy.exists
            if ivy.is_bool_dtype(dtype):
                assert_fn = ivy.is_bool_dtype
            if ivy.is_int_dtype(dtype):
                assert_fn = lambda x: not ivy.is_float_dtype(x)

            if assert_fn:
                ivy.utils.assertions.check_all_or_any_fn(
                    *args,
                    fn=lambda x: (
                        assert_fn(x)
                        if ivy.shape(x) == ()
                        else np_frontend.can_cast(
                            x, ivy.as_ivy_dtype(dtype), casting=casting
                        )
                    ),
                    type="all",
                    message="type of input is incompatible with dtype: {}".format(
                        dtype
                    ),
                )


def _assert_scalar(args, dtype):
    if args and dtype:
        assert_fn = None
        if ivy.is_int_dtype(dtype):
            assert_fn = lambda x: type(x) != float
        elif ivy.is_bool_dtype(dtype):
            assert_fn = lambda x: type(x) == bool

        if assert_fn:
            ivy.utils.assertions.check_all_or_any_fn(
                *args,
                fn=assert_fn,
                type="all",
                message="type of input is incompatible with dtype: {}".format(dtype),
            )


# no casting
def _assert_no_array(args, dtype, scalar_check=False, none=False):
    if args:
        first_arg = args[0]
        fn_func = ivy.as_ivy_dtype(dtype) if ivy.exists(dtype) else ivy.dtype(first_arg)
        assert_fn = lambda x: ivy.dtype(x) == fn_func
        if scalar_check:
            assert_fn = lambda x: (
                ivy.dtype(x) == fn_func
                if ivy.shape(x) != ()
                else _casting_no_special_case(ivy.dtype(x), fn_func, none)
            )
        ivy.utils.assertions.check_all_or_any_fn(
            *args,
            fn=assert_fn,
            type="all",
            message="type of input is incompatible with dtype: {}".format(dtype),
        )


def _casting_no_special_case(dtype1, dtype, none=False):
    if dtype == "float16":
        allowed_dtypes = ["float32", "float64"]
        if not none:
            allowed_dtypes += ["float16"]
        return dtype1 in allowed_dtypes
    if dtype in ["int8", "uint8"]:
        if none:
            return ivy.is_int_dtype(dtype1) and dtype1 not in ["int8", "uint8"]
        return ivy.is_int_dtype(dtype1)
    return dtype1 == dtype


def _assert_no_scalar(args, dtype, none=False):
    if args:
        first_arg = args[0]
        ivy.utils.assertions.check_all_or_any_fn(
            *args,
            fn=lambda x: type(x) == type(first_arg),
            type="all",
            message="type of input is incompatible with dtype {}".format(dtype),
        )
        if dtype:
            if ivy.is_int_dtype(dtype):
                check_dtype = int
            elif ivy.is_float_dtype(dtype):
                check_dtype = float
            else:
                check_dtype = bool
            ivy.utils.assertions.check_equal(
                type(args[0]),
                check_dtype,
                message="type of input is incompatible with dtype {}".format(dtype),
            )
            if ivy.as_ivy_dtype(dtype) not in ["float64", "int8", "int64", "uint8"]:
                if type(args[0]) == int:
                    ivy.utils.assertions.check_elem_in_list(
                        dtype,
                        ["int16", "int32", "uint16", "uint32", "uint64"],
                        inverse=True,
                    )
                elif type(args[0]) == float:
                    ivy.utils.assertions.check_equal(dtype, "float32", inverse=True)


def handle_numpy_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_numpy_dtype(*args, dtype=None, **kwargs):
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
    _handle_numpy_dtype.handle_numpy_dtype = True
    return _handle_numpy_dtype


def handle_numpy_casting(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_numpy_casting(*args, casting="same_kind", dtype=None, **kwargs):
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
        ivy.utils.assertions.check_elem_in_list(
            casting,
            ["no", "equiv", "safe", "same_kind", "unsafe"],
            message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
        )
        args = list(args)
        args_scalar_idxs = ivy.nested_argwhere(
            args, lambda x: isinstance(x, (int, float, bool))
        )
        args_scalar_to_check = ivy.multi_index_nest(args, args_scalar_idxs)
        args_idxs = ivy.nested_argwhere(args, ivy.is_array)
        args_to_check = ivy.multi_index_nest(args, args_idxs)

        if casting in ["no", "equiv"]:
            none = not dtype
            if none:
                dtype = args_to_check[0].dtype if args_to_check else None
            _assert_no_array(
                args_to_check,
                dtype,
                scalar_check=(args_to_check and args_scalar_to_check),
                none=none,
            )
            _assert_no_scalar(args_scalar_to_check, dtype, none=none)
        elif casting in ["same_kind", "safe"]:
            _assert_array(
                args_to_check,
                dtype,
                scalar_check=(args_to_check and args_scalar_to_check),
                casting=casting,
            )
            _assert_scalar(args_scalar_to_check, dtype)

        if ivy.exists(dtype):
            ivy.map_nest_at_indices(
                args, args_idxs, lambda x: ivy.astype(x, ivy.as_ivy_dtype(dtype))
            )

        return fn(*args, **kwargs)

    _handle_numpy_casting.handle_numpy_casting = True
    return _handle_numpy_casting


def handle_numpy_casting_special(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_numpy_casting_special(*args, casting="same_kind", dtype=None, **kwargs):
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
        ivy.utils.assertions.check_elem_in_list(
            casting,
            ["no", "equiv", "safe", "same_kind", "unsafe"],
            message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
        )
        if ivy.exists(dtype):
            ivy.utils.assertions.check_equal(
                ivy.as_ivy_dtype(dtype),
                "bool",
                message="output is compatible with bool only",
            )

        return fn(*args, **kwargs)

    _handle_numpy_casting_special.handle_numpy_casting_special = True
    return _handle_numpy_casting_special


def _numpy_frontend_to_ivy(x: Any) -> Any:
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    else:
        return x


def _ivy_to_numpy(x: Any) -> Any:
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = np_frontend.ndarray(x, _init_overload=True)
        return a
    else:
        return x


def _ivy_to_numpy_order_F(x: Any) -> Any:
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = np_frontend.ndarray(
            0, order="F"
        )  # TODO Find better initialisation workaround
        a.ivy_array = x
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
    elif isinstance(x, np_frontend.ndarray):
        if x._f_contiguous:
            return False
        else:
            return True
    else:
        return None


def _set_order(args, order):
    ivy.utils.assertions.check_elem_in_list(
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
    def _inputs_to_ivy_arrays_np(*args, **kwargs):
        """
        Convert `ndarray` into `ivy.Array` instances.

        Convert all `ndarray` instances in both the positional and keyword arguments
        into `ivy.Array` instances, and then calls the function with the updated
        arguments.

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
        # convert all arrays in the inputs to ivy.Array instances
        ivy_args = ivy.nested_map(args, _to_ivy_array, include_derived={tuple: True})
        ivy_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}
        )
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_arrays_np.inputs_to_ivy_arrays = True
    return _inputs_to_ivy_arrays_np


def outputs_to_numpy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_numpy_arrays(*args, order="K", **kwargs):
        """
        Convert `ivy.Array` into `ndarray` instances.

        Call the function, and then converts all `ivy.Array` instances
        returned by the function into `ndarray` instances.

        The return of the function, with ivy arrays as numpy arrays.
        """
        # handle order and call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        set_default_dtype = False
        if not ("dtype" in kwargs and ivy.exists(kwargs["dtype"])) and any(
            [not (ivy.is_array(i) or hasattr(i, "ivy_array")) for i in args]
        ):
            (
                ivy.set_default_int_dtype("int64")
                if platform.system() != "Windows"
                else ivy.set_default_int_dtype("int32")
            )
            ivy.set_default_float_dtype("float64")
            set_default_dtype = True
        if contains_order:
            if len(args) >= (order_pos + 1):
                order = args[order_pos]
                args = args[:-1]
            order = _set_order(args, order)
            try:
                ret = fn(*args, order=order, **kwargs)
            finally:
                if set_default_dtype:
                    ivy.unset_default_int_dtype()
                    ivy.unset_default_float_dtype()
        else:
            try:
                ret = fn(*args, **kwargs)
            finally:
                if set_default_dtype:
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
    _outputs_to_numpy_arrays.outputs_to_numpy_arrays = True
    return _outputs_to_numpy_arrays


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wrap `fn` so it receives and returns `ivy.Array` instances.

    Wrap `fn` so that input arrays are all converted to `ivy.Array`
    instances and return arrays are all converted to `ndarray`
    instances.
    """
    return outputs_to_numpy_arrays(inputs_to_ivy_arrays(fn))


def from_zero_dim_arrays_to_scalar(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _from_zero_dim_arrays_to_scalar(*args, **kwargs):
        """
        Convert 0 dimensional arrays to float numbers.

        Call the function, and then converts all 0 dimensional array instances in the
        function to float numbers if out argument is not provided.

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
                    raise ivy.utils.exceptions.IvyException(
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
                        raise ivy.utils.exceptions.IvyException(
                            f"Casting to {ivy.dtype(data)} is unsupported"
                        )
        return ret

    _from_zero_dim_arrays_to_scalar.from_zero_dim_arrays_to_scalar = True
    return _from_zero_dim_arrays_to_scalar


def _count_operands(subscript):
    if "->" in subscript:
        input_subscript, output_index = subscript.split("->")
    else:
        input_subscript = subscript
    return len(input_subscript.split(","))


def handle_numpy_out(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_numpy_out(*args, **kwargs):
        if "out" not in kwargs:
            keys = list(inspect.signature(fn).parameters.keys())
            if fn.__name__ == "einsum":
                out_pos = 1 + _count_operands(args[0])
            else:
                out_pos = keys.index("out")
            kwargs = {
                **dict(
                    zip(
                        keys[keys.index("out") :],
                        args[out_pos:],
                    )
                ),
                **kwargs,
            }
            args = args[:out_pos]
        if "out" in kwargs:
            out = kwargs["out"]
            if ivy.exists(out) and not ivy.nested_any(
                out, lambda x: isinstance(x, np_frontend.ndarray)
            ):
                raise ivy.utils.exceptions.IvyException(
                    "Out argument must be an ivy.frontends.numpy.ndarray object"
                )
        return fn(*args, **kwargs)

    _handle_numpy_out.handle_numpy_out = True
    return _handle_numpy_out
