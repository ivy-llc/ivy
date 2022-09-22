# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


@st.composite
def where(draw):
    _, values = draw(helpers.dtype_and_values(available_dtypes=("bool",)))
    return draw(st.just(values) | st.just(True))


@st.composite
def dtype_x_bounded_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=max(len(shape) - 1, 0)))
    return dtype, x, axis


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
def handle_where_and_array_bools(
    where, input_dtype=None, as_variable=None, native_array=None
):
    where_array = isinstance(where, list)
    if where_array:
        where = np.asarray(where, dtype=np.bool_)
        if ivy.exists(input_dtype):
            try:
                input_dtype += ["bool"]
            except TypeError:
                input_dtype = [input_dtype, "bool"]
        if ivy.exists(as_variable):
            try:
                as_variable += [False]
            except TypeError:
                as_variable = [as_variable, False]
        if ivy.exists(native_array):
            try:
                native_array += [False]
            except TypeError:
                native_array = [native_array, False]
    return where
