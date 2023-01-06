import pickle
import numpy as np
import os

from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# Tests #
# ------#

# pickling array test to str
@given(
    array_shape=helpers.lists(
        arg=helpers.ints(min_value=0, max_value=5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[0, 5],
    ),
    dtype=st.sampled_from(list(ivy_np.valid_dtypes)),
    data=st.data(),
)
def test_pickle_to_string(array_shape, dtype, data, device, fw):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    x = ivy.array(x, dtype=dtype, device=device)

    pickled_arr = pickle.dumps(x)
    unpickled_arr = pickle.loads(pickled_arr)

    # check for equality
    assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(unpickled_arr))


# pickling array test to disk
@given(
    array_shape=helpers.lists(
        arg=helpers.ints(min_value=0, max_value=5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[0, 5],
    ),
    dtype=st.sampled_from(list(ivy_np.valid_dtypes)),
    data=st.data(),
)
def test_pickle_to_and_from_disk(array_shape, dtype, data, device, fw):
    # smoke this for torch
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=dtype))
    x = ivy.array(x, dtype=dtype, device=device)

    save_filepath = "ivy_array.pickle"
    pickle.dump(x, open(save_filepath, "wb"))

    assert os.path.exists(save_filepath)

    unpickled_arr = pickle.load(open(save_filepath, "rb"))

    os.remove(save_filepath)

    # check for equality
    assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(unpickled_arr))
