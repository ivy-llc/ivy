# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.optional_get_element",
    dtype_and_x=helpers.dtype_and_values(
        dtype=[
            'int8', 'int16', 'int32', 'complex64', 'complex128', 'bool', 'float16', 'float32', 'float64', 'string', 'uint8'
        ],
        min_value=-100,
        max_value=100,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=5,
        allow_nan=True,
    ),
)
def test_optional_get_element(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )
