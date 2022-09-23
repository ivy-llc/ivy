# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# flatten
@handle_cmd_line_args
@given(
    # TODO: include more args
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
)
def test_numpy_ma_flatten(
    dtype_x,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={
            "data": x,
            "dtype": input_dtype,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        fw=fw,
        frontend="numpy",
        class_name="ma.MaskedArray",
        method_name="flatten",
    )
