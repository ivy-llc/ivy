# global
import numpy as np

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.asmatrix",
    arr=helpers.dtype_and_values(min_num_dims=2, max_num_dims=2),
)
def test_numpy_asmatrix(arr):
    dtype, x = arr
    ret = np_frontend.asmatrix(x[0])
    ret_gt = np.asmatrix(x[0])
    assert ret.shape == ret_gt.shape
    assert ivy.all(ivy.flatten(ret._data) == np.ravel(ret_gt))
