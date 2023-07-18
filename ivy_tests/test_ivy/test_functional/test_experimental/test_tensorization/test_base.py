# global
from hypothesis import reproduce_failure

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test

# Todo fix the Array method


# tensor_to_vec
@reproduce_failure("6.81.2", b"AXicY2AAAkYGCGBEY0MAAABzAAU=")
@handle_test(
    fn_tree="functional.ivy.experimental.tensors.tensor_to_vec",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_tensor_to_vec(*, dtype_values, test_flags, backend_fw, fn_name, on_device):
    x_dtype, x = dtype_values
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=x_dtype,
        x=x,
    )
