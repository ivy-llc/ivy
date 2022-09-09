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


# noinspection PyShadowingNames
def _test_frontend_function_ignoring_unitialized(*args, **kwargs):
    where = kwargs["where"]
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
    helpers.value_test(ret_np_flat=ret_flat, ret_from_np_flat=frontend_ret_flat)


# noinspection PyShadowingNames
def test_frontend_function(*args, where=None, **kwargs):
    if not ivy.exists(where):
        helpers.test_frontend_function(*args, **kwargs)
    kwargs["where"] = where
    if "out" in kwargs and kwargs["out"] is None:
        _test_frontend_function_ignoring_unitialized(*args, **kwargs)
    helpers.test_frontend_function(*args, **kwargs)


# noinspection PyShadowingNames
def handle_where_and_array_bools(
    where, input_dtype=None, as_variable=None, native_array=None
):
    where_array = isinstance(where, list)
    if where_array:
        where = np.asarray(where, dtype=np.bool)
        if ivy.exists(input_dtype):
            input_dtype += ["bool"]
        if ivy.exists(as_variable):
            as_variable += [False]
        if ivy.exists(native_array):
            native_array += [False]
    return where
