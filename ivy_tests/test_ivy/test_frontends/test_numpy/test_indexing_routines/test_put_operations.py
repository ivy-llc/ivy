from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)
import ivy.functional.frontends.numpy as np_frontend


@handle_frontend_test(
    fn_tree="put",
    dtype_x_indices_values=helpers.array_indices_values(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        values_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_numpy_put(
    *,
    dtype_x_indices_values,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    backend_fw,
):
    dtypes, x, indices, values, _ = dtype_x_indices_values
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=x,
        indices=indices,
        values=values,
    )
