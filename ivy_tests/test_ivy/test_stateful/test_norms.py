"""Collection of tests for normalization layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
from ivy.container import Container
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# layer norm
@given(
    x_n_ns_n_target=st.sampled_from(
        [
            (
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [3],
                [[-1.2247356, 0.0, 1.2247356], [-1.2247356, 0.0, 1.2247356]],
            ),
        ]
    ),
    with_v=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
)
def test_layer_norm_layer(
    x_n_ns_n_target, with_v, dtype, as_variable, device, compile_graph, call
):
    # smoke test
    if dtype == "float16":
        return
    x, normalized_shape, target = x_n_ns_n_target
    x = ivy.array(x, dtype=dtype, device=device)
    target = ivy.array(target, dtype=dtype, device=device)
    if as_variable:
        x = ivy.variable(x)
        target = ivy.variable(target)
    if with_v:
        v = Container(
            {
                "scale": ivy.variable(ivy.ones(normalized_shape)),
                "offset": ivy.variable(ivy.zeros(normalized_shape)),
            }
        )
    else:
        v = None
    norm_layer = ivy.LayerNorm(normalized_shape, device=device, v=v)
    ret = norm_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    if not with_v:
        return
    assert np.allclose(call(norm_layer, x), ivy.to_numpy(target))
    # compilation test
    if call in [helpers.torch_call]:
        # this is not a backend implemented function
        return
