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
def handle_where_and_array_bools(where, input_dtype, as_variable, native_array):
    if isinstance(where, list):
        input_dtype += ["bool"]
        return where, as_variable + [False], native_array + [False]
    return where, as_variable, native_array
