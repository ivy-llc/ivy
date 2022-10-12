# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


@st.composite
def where(draw):
    _, values = draw(helpers.dtype_and_values(dtype=["bool"]))
    return draw(st.just(values) | st.just(True))


@st.composite
def _get_safe_casting_dtype(draw, *, dtypes):
    target_dtype = dtypes[0]
    for dtype in dtypes[1:]:
        if ivy.can_cast(target_dtype, dtype):
            target_dtype = dtype
    if ivy.is_float_dtype(target_dtype):
        dtype = draw(st.sampled_from(["float64", None]))
    elif ivy.is_uint_dtype(target_dtype):
        dtype = draw(st.sampled_from(["uint64", None]))
    elif ivy.is_int_dtype(target_dtype):
        dtype = draw(st.sampled_from(["int64", None]))
    else:
        dtype = draw(st.sampled_from(["bool", None]))
    return dtype


@st.composite
def dtypes_values_casting_dtype(
    draw,
    *,
    arr_func,
    get_dtypes_kind="valid",
    get_dtypes_index=0,
    get_dtypes_none=True,
    get_dtypes_key=None,
    special=False,
):
    dtypes, values = [], []
    casting = draw(st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"]))
    for idx, func in enumerate(arr_func):
        typ, val = draw(func())
        if casting in ["no", "equiv"] and idx > 0:
            dtypes.append(dtypes[0])
        else:
            dtypes += typ if isinstance(typ, list) else [typ]
        values += val if isinstance(val, list) else [val]

    if special:
        dtype = draw(st.sampled_from(["bool", None]))
    elif casting in ["no", "equiv"]:
        dtype = draw(st.just(None))
    elif casting in ["safe", "same_kind"]:
        dtype = draw(_get_safe_casting_dtype(dtypes=dtypes))
    else:
        dtype = draw(
            helpers.get_dtypes(
                get_dtypes_kind,
                index=get_dtypes_index,
                full=False,
                none=get_dtypes_none,
                key=get_dtypes_key,
            )
        )
    return dtypes, values, casting, dtype


@st.composite
def dtype_x_bounded_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=max(len(shape) - 1, 0)))
    return dtype, x, axis


@st.composite
def _array_and_axes_permute_helper(
    draw,
    *,
    min_num_dims,
    max_num_dims,
    min_dim_size,
    max_dim_size,
    allow_none=False,
):
    """Returns array, its dtype and either the random permutation of its axes or None.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_num_dims
        minimum number of array dimensions
    max_num_dims
        maximum number of array dimensions
    min_dim_size
        minimum size of the dimension
    max_dim_size
        maximum size of the dimension

    Returns
    -------
    A strategy that draws an array, its dtype and axes (or None).
    """
    shape = draw(
        helpers.get_shape(
            allow_none=allow_none,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    dtype = draw(helpers.array_dtypes(num_arrays=1))[0]
    array = draw(helpers.array_values(dtype=dtype, shape=shape))
    axes = draw(
        st.one_of(
            st.none(),
            helpers.get_axis(
                shape=shape,
                allow_neg=False,
                allow_none=False,
                sorted=False,
                unique=True,
                min_size=len(shape),
                max_size=len(shape),
                force_tuple=True,
                force_int=False,
            ),
        ).filter(lambda x: x != tuple(range(len(shape))))
    )
    return (array, dtype, axes)


# noinspection PyShadowingNames
def _test_frontend_function_ignoring_unitialized(*args, **kwargs):
    where = kwargs["where"]
    kwargs["test_values"] = False
    values = helpers.test_frontend_function(*args, **kwargs)
    if values is None:
        return
    ret, frontend_ret = values
    ret_flat = [
        np.where(where, x, np.zeros_like(x))
        for x in helpers.flatten_fw(ret=ret, fw=kwargs["fw"])
    ]
    frontend_ret_flat = [
        np.where(where, x, np.zeros_like(x))
        for x in helpers.flatten_fw(ret=frontend_ret, fw=kwargs["frontend"])
    ]
    helpers.value_test(ret_np_flat=ret_flat, ret_np_from_gt_flat=frontend_ret_flat)


# noinspection PyShadowingNames
def test_frontend_function(*args, where=None, **kwargs):
    if not ivy.exists(where):
        helpers.test_frontend_function(*args, **kwargs)
    else:
        kwargs["where"] = where
        if "out" in kwargs and kwargs["out"] is None:
            _test_frontend_function_ignoring_unitialized(*args, **kwargs)
        else:
            helpers.test_frontend_function(*args, **kwargs)


# noinspection PyShadowingNames
def _test_frontend_array_instance_method_ignoring_unitialized(*args, **kwargs):
    where = kwargs["where"]
    kwargs["test_values"] = False
    values = helpers.test_frontend_array_instance_method(*args, **kwargs)
    if values is None:
        return
    ret, frontend_ret = values
    ret_flat = [
        np.where(where, x, np.zeros_like(x))
        for x in helpers.flatten_fw(ret=ret, fw=kwargs["fw"])
    ]
    frontend_ret_flat = [
        np.where(where, x, np.zeros_like(x))
        for x in helpers.flatten_fw(ret=frontend_ret, fw=kwargs["frontend"])
    ]
    helpers.value_test(ret_np_flat=ret_flat, ret_np_from_gt_flat=frontend_ret_flat)


# noinspection PyShadowingNames
def test_frontend_array_instance_method(*args, where=None, **kwargs):
    if not ivy.exists(where):
        helpers.test_frontend_array_instance_method(*args, **kwargs)
    else:
        kwargs["where"] = where
        if "out" in kwargs and kwargs["out"] is None:
            _test_frontend_array_instance_method_ignoring_unitialized(*args, **kwargs)
        else:
            helpers.test_frontend_array_instance_method(*args, **kwargs)


# noinspection PyShadowingNames
def handle_where_and_array_bools(where, input_dtype, as_variable, native_array):
    if isinstance(where, list):
        input_dtype += ["bool"]
        return where, as_variable + [False], native_array + [False]
    return where, as_variable, native_array


def handle_dtype_and_casting(
    *,
    dtypes,
    get_dtypes_kind="valid",
    get_dtypes_index=0,
    get_dtypes_none=True,
    get_dtypes_key=None,
):
    casting = st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"])
    if casting in ["no", "equiv"]:
        dtype = dtypes[0]
        dtypes = [dtype for x in dtypes]
        return dtype, dtypes, casting
    dtype = helpers.get_dtypes(
        get_dtypes_kind,
        index=get_dtypes_index,
        full=False,
        none=get_dtypes_none,
        key=get_dtypes_key,
    )
    if casting in ["safe", "same_kind"]:
        while not ivy.all([ivy.can_cast(x, dtype) for x in dtypes]):
            dtype = helpers.get_dtypes(
                get_dtypes_kind,
                index=get_dtypes_index,
                full=False,
                none=get_dtypes_none,
                key=get_dtypes_key,
            )
    return dtype, dtypes, casting
