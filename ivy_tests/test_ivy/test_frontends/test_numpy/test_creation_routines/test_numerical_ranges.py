# global
import numpy as np
from numpy import mgrid as np_mgrid, ogrid as np_ogrid
from hypothesis import given, strategies as st

# local
from ivy.functional.frontends.numpy import mgrid, ogrid
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# helpers
@st.composite
def _get_range_for_grid(draw):
    start = draw(st.booleans())
    step = draw(st.booleans())
    if start:
        start = draw(helpers.ints(min_value=-25, max_value=25))
        stop = draw(st.booleans())
        if stop:
            stop = draw(helpers.ints(min_value=30, max_value=100))
        else:
            stop = None
    else:
        start = None
        stop = draw(helpers.ints(min_value=30, max_value=100))
    if step:
        step = draw(helpers.ints(min_value=1, max_value=5))
        return start, stop, step
    return start, stop, None


@st.composite
def _get_dtype_and_range(draw):
    dim = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    start = draw(
        helpers.array_values(dtype=dtype, shape=(dim,), min_value=-50, max_value=0)
    )
    stop = draw(
        helpers.array_values(dtype=dtype, shape=(dim,), min_value=1, max_value=50)
    )
    return dtype, start, stop


# arange
@handle_cmd_line_args
@given(
    start=helpers.ints(min_value=-50, max_value=0),
    stop=helpers.ints(min_value=1, max_value=50),
    step=helpers.ints(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("numeric", full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.arange"
    ),
)
def test_numpy_arange(
    start,
    stop,
    step,
    dtype,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_tree="arange",
        start=start,
        stop=stop,
        step=step,
        dtype=dtype,
    )


# linspace
@handle_cmd_line_args
@given(
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linspace"
    ),
)
def test_numpy_linspace(
    dtype_start_stop,
    num,
    axis,
    num_positional_args,
    fw,
):
    dtype, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=[dtype, dtype],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_tree="linspace",
        start=np.asarray(start, dtype=dtype),
        stop=np.asarray(stop, dtype=dtype),
        num=num,
        endpoint=True,
        retstep=False,
        dtype=dtype,
        axis=axis,
    )


# logspace
@handle_cmd_line_args
@given(
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=5, max_value=50),
    base=helpers.ints(min_value=2, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.logspace"
    ),
)
def test_numpy_logspace(
    dtype_start_stop,
    num,
    base,
    axis,
    num_positional_args,
    fw,
):
    dtype, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=[dtype, dtype],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_tree="logspace",
        rtol=1e-01,
        start=np.asarray(start, dtype=dtype),
        stop=np.asarray(stop, dtype=dtype),
        num=num,
        endpoint=True,
        base=base,
        dtype=dtype,
        axis=axis,
    )


# meshgrid
@handle_cmd_line_args
@given(
    dtype_and_arrays=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        min_dim_size=1,
        shared_dtype=True,
    ),
    copy=st.booleans(),
    sparse=st.booleans(),
    indexing=st.sampled_from(["xy", "ij"]),
)
def test_numpy_meshgrid(
    dtype_and_arrays,
    copy,
    sparse,
    indexing,
    fw,
):
    input_dtypes, arrays = dtype_and_arrays
    kw = {}
    i = 0
    for x_ in arrays:
        kw["x{}".format(i)] = np.asarray(x_, dtype=input_dtypes[0])
        i += 1
    num_positional_args = len(arrays)

    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_tree="meshgrid",
        **kw,
        copy=copy,
        sparse=sparse,
        indexing=indexing,
    )


# mgrid
@handle_cmd_line_args
@given(range=_get_range_for_grid())
def test_numpy_mgrid(
    range,
):
    start, stop, step = range
    if start and stop and step:
        ret = mgrid[start:stop:step]
        ret_np = np_mgrid[start:stop:step]
    elif start and step:
        ret = mgrid[start::step]
        ret_np = np_mgrid[start::step]
    elif stop and step:
        ret = mgrid[:stop:step]
        ret_np = np_mgrid[:stop:step]
    elif start and stop:
        ret = mgrid[start:stop]
        ret_np = np_mgrid[start:stop]
    elif start:
        ret = mgrid[start:]
        ret_np = np_mgrid[start:]
    else:
        ret = mgrid[:stop]
        ret_np = np_mgrid[:stop]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_np = helpers.flatten_and_to_np(ret=ret_np)
    helpers.value_test(ret_np_flat=ret, ret_np_from_gt_flat=ret_np, rtol=1e-03)


# ogrid
@handle_cmd_line_args
@given(range=_get_range_for_grid())
def test_numpy_ogrid(
    range,
):
    start, stop, step = range
    if start and stop and step:
        ret = ogrid[start:stop:step]
        ret_np = np_ogrid[start:stop:step]
    elif start and step:
        ret = ogrid[start::step]
        ret_np = np_ogrid[start::step]
    elif stop and step:
        ret = ogrid[:stop:step]
        ret_np = np_ogrid[:stop:step]
    elif start and stop:
        ret = ogrid[start:stop]
        ret_np = np_ogrid[start:stop]
    elif start:
        ret = ogrid[start:]
        ret_np = np_ogrid[start:]
    else:
        ret = ogrid[:stop]
        ret_np = np_ogrid[:stop]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_np = helpers.flatten_and_to_np(ret=ret_np)
    helpers.value_test(ret_np_flat=ret, ret_np_from_gt_flat=ret_np, rtol=1e-03)
