import pickle
import numpy as np
import os

from hypothesis import given, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# Tests #
# ------#

# pickling array test to str
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=5,
    ),
)
def test_pickle_to_string(dtype_and_x, on_device):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    x = ivy.array(x[0], dtype=input_dtype[0], device=on_device)

    pickled_arr = pickle.dumps(x)
    unpickled_arr = pickle.loads(pickled_arr)

    # check for equality
    assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(unpickled_arr))


# pickling array test to disk
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=5,
    ),
)
def test_pickle_to_and_from_disk(dtype_and_x, on_device):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    x = ivy.array(x[0], dtype=input_dtype[0], device=on_device)

    save_filepath = "ivy_array.pickle"
    pickle.dump(x, open(save_filepath, "wb"))

    assert os.path.exists(save_filepath)

    unpickled_arr = pickle.load(open(save_filepath, "rb"))

    os.remove(save_filepath)

    # check for equality
    assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(unpickled_arr))
