# global
import numpy as np
from hypothesis import given

# local
import ivy.functional.frontends.numpy as np_frontend
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test, BackendHandler


@handle_frontend_test(
    fn_tree="numpy.asmatrix",
    arr=helpers.dtype_and_values(min_num_dims=2, max_num_dims=2),
)
def test_numpy_asmatrix(arr, backend_fw):
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        dtype, x = arr
        ret = np_frontend.asmatrix(x[0])
        ret_gt = np.asmatrix(x[0])
        assert ret.shape == ret_gt.shape
        assert ivy_backend.all(ivy_backend.flatten(ret._data) == np.ravel(ret_gt))


@handle_frontend_test(
    fn_tree="numpy.asscalar",
    arr=helpers.array_values(dtype=helpers.get_dtypes("numeric"), shape=1),
)
def test_numpy_asscalar(arr: np.ndarray):
    ret_1 = arr.item()
    ret_2 = np_frontend.asscalar(arr)
    assert ret_1 == ret_2


# asanyarray
@handle_frontend_test
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.asanyarray"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_asanyarray(
        dtype_and_x,
        as_variable,
        num_positional_args,
        native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="asanyarray",
        a=x,
        dtype=input_dtype,
    )
