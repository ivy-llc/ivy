"""Collection of tests for unified neural network activations."""

# global
#import pytest
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy
#import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# GELU
@given(
    x=st.lists(st.floats()),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
)
def test_gelu(x, dtype ,with_out, as_variable, native_array, fw):
    if dtype in ivy.invalid_dtypes:
        return  # invalid dtype
    if dtype == "float16" and fw == "torch":
        return  # torch does not support float16 for gelu
    x = ivy.array(x, dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            return  # only floating point variables are supported
        if with_out:
            return  # variables do not support out argument
        x = ivy.variable(x)
    if native_array:
        x = x.data
    ret = ivy.gelu(x)
    out = ret
    if with_out:
        if as_variable:
            out = ivy.variable(out)
        if native_array:
            out = out.data
        ret = ivy.gelu(x, out=out)
        if not native_array:
            assert ret is out
        if fw in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)
    # value test
    if dtype == "bfloat16":
        return  # bfloat16 is not supported by numpy
    assert np.allclose(
        np.nan_to_num(ivy.to_numpy(ret)), np.nan_to_num(ivy_np.gelu(ivy.to_numpy(x)))
    )


# GEGLU
@given(
    bs_oc_target=st.sampled_from(
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
    ]),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
)
def test_geglu(bs_oc_target, dtype, device, compile_graph, call):
    # smoke test
    batch_shape, output_channels, target = bs_oc_target
    x = ivy.asarray(
        ivy.linspace(
            ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels * 2
        ),
        "float32",
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
