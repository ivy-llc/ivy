# global

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.numpy.ufunc"


# at
@handle_frontend_method(
    method_name="at",
    init_tree="numpy.ufunc",
    class_tree=CLASS_TREE,
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    )
)
def test_numpy_ufunc_at(
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    frontend_method_data,
    frontend,
):
    input_dtype, x = dtype_and_x
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="numeric",
    )
    np_frontend_helpers.test_frontend_method(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        method_num_positional_args=method_num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        x=x[0],
        dtype=dtype,
    )
