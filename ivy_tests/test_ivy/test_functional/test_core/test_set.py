# global
import numpy as np
import pytest
from hypothesis import strategies as st, given

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# unique_values
@given(array_shape=helpers.lists(
    st.integers(2, 3),
    min_size="num_dims",
    max_size="num_dims",
    size_bounds=[1, 3]),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    repeats=st.integers(2,5),
    with_out=st.booleans())
def test_unique_values(array_shape,
                       dtype,
                       as_variable,
                       repeats,
                       with_out,
                       fw,
                       device):
    if fw == "torch" and ("int" in dtype or "16" in dtype):
        return

    shape = tuple(array_shape)
    arr = np.random.uniform(size=shape).astype(dtype).flatten()
    arr = ivy.asarray(np.repeat(arr, repeats=repeats))

    if as_variable:
        arr = ivy.variable(arr)

    gt = ivy.unique_values(arr)

    # create dummy out so that it is broadcastable to gt
    out = ivy.zeros(ivy.shape(gt)) if with_out else None

    # do the operation
    res = ivy.unique_values(arr, out=out)

    assert np.allclose(ivy.to_numpy(res), ivy.to_numpy(gt))

    if with_out:
        # match the values of res and out
        assert np.allclose(ivy.to_numpy(res), ivy.to_numpy(out))

        if ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            return

        # native arrays should be the same object
        assert res.data is out.data

# Still to Add #
# ---------------#

# unique_all
# unique_counts
# unique_inverse
