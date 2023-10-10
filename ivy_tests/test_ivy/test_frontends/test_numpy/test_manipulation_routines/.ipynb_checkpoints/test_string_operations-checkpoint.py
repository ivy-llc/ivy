#global
import numpy as np
from hypothesis import @handle_frontend_test, strategies as st


#local
import ivy_tests.tests_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

#add
@handle_frontend_test(
    fn_tree="numpy.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_add(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    out
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=x[0],
        y=x[1],
        if "out" in kwargs and kwargs["out"] is None:
            _test_frontend_function_ignoring_uninitialized(*args, **kwargs)
            return
        else:
            helpers.test_frontend_function(*args, **kwargs)
    )