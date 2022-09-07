# global
# import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# helpers
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
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
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
        available_dtypes=ivy_np.valid_float_dtypes,
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


# TODO
# mgrid
# ogrid
