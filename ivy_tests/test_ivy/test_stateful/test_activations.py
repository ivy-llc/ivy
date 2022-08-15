"""Collection of tests for unified neural network activations."""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# GELU
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
        dtype="float32",
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
    if ivy.current_backend_str() == "torch":
        # pytest scripting does not **kwargs
        return


# GEGLU
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
                        0.02189754,
                        0.04893785,
                        0.08134944,
                        0.11933776,
                        0.16308454,
                        0.21274757,
                        0.26846102,
                        0.3303356,
                        0.39845937,
                    ],
                    [
                        0.0,
                        0.02189754,
                        0.04893785,
                        0.08134944,
                        0.11933776,
                        0.16308454,
                        0.21274757,
                        0.26846102,
                        0.3303356,
                        0.39845937,
                    ],
                ]
            ],
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_geglu(bs_oc_target, dtype, tensor_fn, device, compile_graph, call):
    # smoke test
    batch_shape, output_channels, target = bs_oc_target
    x = ivy.asarray(
        ivy.linspace(
            ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels * 2
        ),
        dtype="float32",
    )
    geglu_layer = ivy.GEGLU()

    # return
    ret = geglu_layer(x)

    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [output_channels])
    # value test
    assert np.allclose(call(geglu_layer, x), np.array(target))
