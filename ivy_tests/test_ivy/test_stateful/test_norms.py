"""Collection of tests for normalization layers."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# layer norm
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
)
def test_layer_norm_layer(
    array_shape, dtype, device, call
):
    if dtype in ['float16']:
        return
    # smoke test
    x = ivy.asarray(np.random.uniform(size=tuple(array_shape)).astype(dtype))
    norm_layer = ivy.LayerNorm(x.shape, device=device)
    ret = norm_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(norm_layer, x), ivy.to_numpy(ret))
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        return
