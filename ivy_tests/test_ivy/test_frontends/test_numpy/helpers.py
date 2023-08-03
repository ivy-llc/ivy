# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.frontends.numpy as np_frontend


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
    """
    Return array, its dtype and either the random permutation of its axes or None.

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
    dtype = draw(helpers.array_dtypes(num_arrays=1))
    array = draw(helpers.array_values(dtype=dtype[0], shape=shape))
    axes = draw(
        st.one_of(
            st.none(),
            helpers.get_axis(
                shape=shape,
                allow_neg=False,
                allow_none=False,
                sort_values=False,
                unique=True,
                min_size=len(shape),
                max_size=len(shape),
                force_tuple=True,
                force_int=False,
            ),
        ).filter(lambda x: x != tuple(range(len(shape))))
    )
    return array, dtype, axes


@st.composite
def where(draw, *, shape=None):
    if shape is None:
        _, values = draw(helpers.dtype_and_values(dtype=["bool"]))
    else:
        _, values = draw(helpers.dtype_and_values(dtype=["bool"], shape=shape))
    return draw(st.just(values) | st.just(True))


# noinspection PyShadowingNames
def _test_frontend_function_ignoring_uninitialized(*args, **kwargs):
    # TODO: this is a hack to get around, but not sure if it is efficient way to do it.
    where = kwargs["where"]
    if kwargs["frontend"] == "numpy":
        kwargs["where"] = True
    else:
        kwargs["where"] = None
    kwargs["test_values"] = False
    values = helpers.test_frontend_function(*args, **kwargs)
    if values is None:
        return
    ret, frontend_ret = values
    # set backend to frontend to flatten the frontend array
    ivy.set_backend(kwargs["frontend"])
    try:
        # get flattened arrays from returned value
        if ivy.isscalar(frontend_ret):
            frontend_ret_np_flat = [np.asarray(frontend_ret)]
        else:
            if not isinstance(frontend_ret, tuple):
                frontend_ret = (frontend_ret,)
            frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_native_array)
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
            frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    except Exception as e:
        ivy.previous_backend()
        raise e
    # set backend back to original
    ivy.previous_backend()

    # get flattened arrays from returned value
    ret_np_flat = _flatten_frontend_return(ret=ret)

    # handling where size
    where = np.asarray(where)
    if where.ndim == 0:
        where = np.array([where])
    elif where.ndim > 1:
        where = where.flatten()
    # handling ret size

    first_el = ret_np_flat[0]
    # change where to match the shape of the first element of ret_np_flat
    if first_el.size == 1:
        where = where[:1]
    else:
        where = np.repeat(where, first_el.size)
        where = where[: first_el.size]
        where = where.reshape(first_el.shape)

    ret_flat = [np.where(where, x, np.zeros_like(x)) for x in ret_np_flat]
    frontend_ret_flat = [
        np.where(where, x, np.zeros_like(x)) for x in frontend_ret_np_flat
    ]
    if "rtol" in kwargs.keys():
        rtol = kwargs["rtol"]
    else:
        rtol = 1e-4
    if "atol" in kwargs.keys():
        atol = kwargs["atol"]
    else:
        atol = 1e-6
    helpers.value_test(
        ret_np_flat=ret_flat,
        ret_np_from_gt_flat=frontend_ret_flat,
        rtol=rtol,
        atol=atol,
    )


def _flatten_frontend_return(*, ret):
    """Flattening the returned frontend value to a list of numpy arrays."""
    current_backend = ivy.current_backend_str()
    if not isinstance(ret, tuple):
        if not ivy.is_ivy_array(ret):
            ret_np_flat = helpers.flatten_frontend_to_np(ret=ret)
        else:
            ret_np_flat = helpers.flatten_fw_and_to_np(ret=ret, fw=current_backend)
    else:
        if any([not ivy.is_ivy_array(x) for x in ret]):
            ret_np_flat = helpers.flatten_frontend_to_np(ret=ret)
        else:
            ret_np_flat = helpers.flatten_fw_and_to_np(ret=ret, fw=current_backend)
    return ret_np_flat


# noinspection PyShadowingNames
def test_frontend_function(*args, where=None, **kwargs):
    if not ivy.exists(where):
        helpers.test_frontend_function(*args, **kwargs)
    else:
        kwargs["where"] = where
        if "out" in kwargs and kwargs["out"] is None:
            _test_frontend_function_ignoring_uninitialized(*args, **kwargs)
            return
        else:
            helpers.test_frontend_function(*args, **kwargs)


# noinspection PyShadowingNames
def handle_where_and_array_bools(where, input_dtype, test_flags):
    if isinstance(where, list) or isinstance(where, tuple):
        where = where[0]
        test_flags.as_variable += [False]
        test_flags.native_arrays += [False]
        input_dtype += ["bool"]
        return where, input_dtype, test_flags
    return where, input_dtype, test_flags


# Casting helper
@st.composite
def _get_safe_casting_dtype(draw, *, dtypes):
    target_dtype = dtypes[0]
    for dtype in dtypes[1:]:
        if np_frontend.can_cast(target_dtype, dtype, casting="safe"):
            target_dtype = dtype
    if ivy.is_float_dtype(target_dtype):
        dtype = draw(st.sampled_from(["float64", None]))
    elif ivy.is_uint_dtype(target_dtype):
        dtype = draw(st.sampled_from(["uint64", None]))
    elif ivy.is_int_dtype(target_dtype):
        dtype = draw(st.sampled_from(["int64", None]))
    elif ivy.is_complex_dtype(target_dtype):
        dtype = draw(st.sampled_from(["complex128", None]))
    else:
        dtype = draw(st.sampled_from(["bool", None]))
    # filter uint64 as not supported by torch backend
    if dtype == "uint64":
        dtype = None
    return dtype


@st.composite
def dtypes_values_casting_dtype(
    draw,
    *,
    arr_func,
    get_dtypes_none=True,
    special=False,
):
    dtypes, values = [], []
    casting = draw(st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"]))
    for func in arr_func:
        typ, val = draw(func())
        dtypes += typ if isinstance(typ, list) else [typ]
        values += val if isinstance(val, list) else [val]

    if casting in ["no", "equiv"] and len(dtypes) > 0:
        dtypes = [dtypes[0]] * len(dtypes)

    if special:
        dtype = draw(st.sampled_from(["bool", None]))
    elif get_dtypes_none:
        dtype = draw(st.sampled_from([None]))
    elif casting in ["no", "equiv"]:
        dtype = draw(st.just(None))
    elif casting in ["safe", "same_kind"]:
        dtype = draw(_get_safe_casting_dtype(dtypes=dtypes))
    else:
        dtype = draw(st.sampled_from([None]))
    return dtypes, values, casting, dtype


# ufunc num_positional_args helper
@st.composite
def get_num_positional_args_ufunc(draw, *, fn_name=None):
    """
    Draws data randomly from numbers between nin and nargs where nin and nargs are
    properties of the given ufunc.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible)
        from a given data-set (ex. list).
    fn_name
        name of the ufunc.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.
    """
    func = getattr(np_frontend, fn_name)
    nin = func.nin
    nargs = func.nargs
    return draw(st.integers(min_value=nin, max_value=nargs))


@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    if dim_size == 1:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
    else:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
    return dtype, vec1, vec2
