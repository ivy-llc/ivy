import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.torch as ivy_torch


# pixel_shuffle
@given(
    dtype_and_x=helpers.dtype_and_values(
    available_dtypes=tuple(ivy_torch.valid_float_dtypes, ivy_torch.valid_int_dtypes)),
    as_variable=helpers.array_bools(),
    with_out=True,
    num_positional_args=helpers.num_positional_args(
    fn_name="functional.frontends.torch.pixel_shuffle"),
    native_array=helpers.array_bools(),
)
def test_torch_pixel_shuffle(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="pixel_shuffle",
        input=np.asarray(x, dtype=input_dtype),
        out=None,
    )
