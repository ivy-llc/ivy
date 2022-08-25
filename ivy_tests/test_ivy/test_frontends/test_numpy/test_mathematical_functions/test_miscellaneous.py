# global
import numpy as np
from hypothesis import given, strategies as st, settings

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

@handle_cmd_line_args
@given(
    x_min_n_max=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, num_arrays=3, shared_dtype=True
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(num_arrays=1),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.clip"
    ),
    native_array=helpers.array_bools(num_arrays=1),
)
@settings(max_examples=50)
def test_numpy_clip(
    x_min_n_max,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    (x_dtype, min_dtype, max_dtype), (x_list, min_val_list, max_val_list) = x_min_n_max
    input_dtype = [x_dtype]
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
        fw="numpy",
        frontend="numpy",
        fn_tree="clip",
        x=np.asarray(x_list, dtype=input_dtype[0]),
        a_min=min_val_list,
        a_max=max_val_list,
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )