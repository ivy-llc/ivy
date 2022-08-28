import numpy as np
from hypothesis import given, strategies as st

import ivy
# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy.current_backend().valid_float_dtypes, min_num_dims=1
    ),
    dtype=st.sampled_from(ivy.current_backend().valid_float_dtypes + (None,)),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.floor"
    ),
)
def test_numpy_floor(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    if type(input_dtype) != list:
        input_dtype = [input_dtype]
    if type(as_variable) != list:
        as_variable = [as_variable]
    if type(native_array) != list:
        native_array = [native_array]

    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="floor",
        x=np.asarray(x, dtype=input_dtype[0]),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )
