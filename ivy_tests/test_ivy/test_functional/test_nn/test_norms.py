"""Collection of tests for unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


@given(
    data=st.data(),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_layer_norm(
    data,
    input_dtype,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    num_positional_args = data.draw(helpers.num_positional_args(fn_name="layer_norm"))
    shape = data.draw(helpers.get_shape(min_num_dims=1))
    x = data.draw(helpers.array_values(dtype=input_dtype, shape=shape))
    normalized_idxs = data.draw(helpers.get_axis(shape=shape))
    scale, offset = tuple(data.draw(helpers.array_values(input_dtype, shape=(2,))))
    epsilon, new_std = tuple(
        data.draw(
            helpers.array_values(input_dtype, shape=(2,), min_value=0, exclude_min=True)
        )
    )
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "layer_norm",
        x=np.asarray(x, dtype=input_dtype),
        normalized_idxs=normalized_idxs,
        epsilon=epsilon,
        scale=scale,
        offset=offset,
        new_std=new_std,
    )
