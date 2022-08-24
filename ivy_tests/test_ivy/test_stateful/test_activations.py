"""Collection of tests for unified neural network activations."""

# global
import pytest
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# GELU
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    approximate=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="GELU.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="GELU._forward"),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={"approximate": approximate},
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"x": np.asarray(x, dtype=input_dtype)},
        fw=fw,
        class_name="GELU",
    )


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
def test_geglu(bs_oc_target, dtype, tensor_fn, device, compile_graph):
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
    assert np.allclose(ivy.to_numpy(geglu_layer(x)), np.array(target))
