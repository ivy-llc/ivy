import pytest
from numbers import Number
import pickle
import numpy as np
import os

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# Tests #
# ------#

# pickling array test to str
@pytest.mark.parametrize("x", [1, [], [1], [1, 2, 3], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_pickle_to_string(x, dtype, tensor_fn, device, call):
    # smoke test
    if (
        (isinstance(x, Number) or len(x) == 0)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    x = tensor_fn(x, dtype, device)
    pickled_arr = pickle.dumps(x)
    unpickled_arr = pickle.loads(pickled_arr)

    # check for equality
    assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(unpickled_arr))


# pickling array test to disk
@pytest.mark.parametrize("x", [1, [], [1], [1, 2, 3], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_pickle_to_and_from_disk(x, dtype, tensor_fn, device, call):
    # smoke test
    if (
        (isinstance(x, Number) or len(x) == 0)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    x = tensor_fn(x, dtype, device)
    save_filepath = "ivy_array.pickle"
    pickle.dump(x, open(save_filepath, "wb"))

    assert os.path.exists(save_filepath)

    unpickled_arr = pickle.load(open(save_filepath, "rb"))

    os.remove(save_filepath)

    # check for equality
    assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(unpickled_arr))
