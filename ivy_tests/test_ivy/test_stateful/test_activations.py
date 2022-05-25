"""Collection of tests for unified neural network activations."""

# global
from hypothesis import given, strategies as st
import numpy as np
import pytest
# local
from ivy_tests.test_ivy import helpers

import ivy
import ivy.functional.backends.numpy as ivy_np

#GELU
@pytest.mark.parametrize(
    "bs_oc_target",
    [
        (
            [1, 2],
            10,
            [
                [
                    [
                        0.0,
                        0.0604706,
                        0.13065024,
                        0.21018247,
                        0.2984952,
                        0.39483193,
                        0.49829122,
                        0.6078729,
                        0.7225253,
                        0.841192,
                    ],
                    [
                        0.0,
                        0.0604706,
                        0.13065024,
                        0.21018247,
                        0.2984952,
                        0.39483193,
                        0.49829122,
                        0.6078729,
                        0.7225253,
                        0.841192,
                    ],
                ]
            ],
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_gelu(bs_oc_target, dtype, tensor_fn, device, compile_graph, call):
    # smoke test
    batch_shape, output_channels, target = bs_oc_target
    x = ivy.asarray(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    gelu_layer = ivy.GELU()

    # return
    ret = gelu_layer(x)

    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [output_channels])
    # value test
    assert np.allclose(call(gelu_layer, x), np.array(target))
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return



@given(
    dtype=st.sampled_from(ivy.float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    class_method=st.booleans(),
)
def test_geglu( dtype, as_variable, with_out,class_method, num_positional_args, native_array, container, fw,):
    helpers.test_array_class(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        class_method,
        fw,
        "GEGLU",
    )
