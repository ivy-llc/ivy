# TODO THIS MUST NOT BE MERGED TO MASTER FOR PLAYGROUND PURPOSES ONLY
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.backend.pipeline import BackendPipeline
import numpy as np

test_flags = FunctionTestFlags(
    ground_truth_backend="tensorflow",
    num_positional_args=1,
    with_out=False,
    instance_method=False,
    as_variable=[False],
    native_arrays=[False],
    container=[False],
    test_gradients=False,
    test_compile=False,
    precision_mode=False,
)

input_dtype, x = ["int8"], [np.array([0], dtype=np.int8)]
BackendPipeline.test_function(
    input_dtypes=input_dtype,
    test_flags=test_flags,
    backend_to_test="numpy",
    fn_name="abs",
    on_device="cpu",
    x=x[0],
)
