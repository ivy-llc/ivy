# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# mean
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes
    ),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes + (None,)),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.mean"
    ),
    native_array=helpers.array_bools(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_numpy_mean(
    *,
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        frontend="numpy",
        fn_name="mean",
        x=np.asarray(x, dtype=input_dtype),
        axis=None,
        keepdims=False,
    )
