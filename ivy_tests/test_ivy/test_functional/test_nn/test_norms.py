"""Collection of tests for unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# # layer norm
# @pytest.mark.parametrize(
#     "x_n_ni_n_s_n_o_n_res",
#     [
#         (
#             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
#             -1,
#             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
#             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
#             [[-0.22473562, 2.0, 6.6742067], [-0.8989425, 5.0, 13.348413]],
#         ),
#     ],
# )
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array])
@given(
    data=st.data(),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 4),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 4),
    native_array=helpers.list_of_length(st.booleans(), 4),
    container=helpers.list_of_length(st.booleans(), 4),
    instance_method=st.booleans(),
)
def test_layer_norm(
    data,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    # smoke test
    shape = data.draw(helpers.get_shape(min_num_dims=1))
    x = data.draw(helpers.array_values(dtype=input_dtype, shape=shape))
    norm_idxs = data.draw(helpers.get_axis(shape=shape))
    scale = data.draw(helpers.array_values(dtype=input_dtype, shape=shape))
    offset = data.draw(helpers.array_values(dtype=input_dtype, shape=shape))
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
        norm_idxs=norm_idxs,
        scale=np.asarray(scale, input_dtype),
        offset=np.asarray(offset, input_dtype),
    )
    # x, norm_idxs, scale, offset, true_res = x_n_ni_n_s_n_o_n_res
    # x = tensor_fn(x, dtype=dtype, device=device)
    # scale = tensor_fn(scale, dtype=dtype, device=device)
    # offset = tensor_fn(offset, dtype=dtype, device=device)
    # true_res = tensor_fn(true_res, dtype=dtype, device=device)
    # ret = ivy.layer_norm(x, norm_idxs, scale=scale, offset=offset)
    # # type test
    # assert ivy.is_ivy_array(ret)
    # # cardinality test
    # assert ret.shape == true_res.shape
    # # value test
    # assert np.allclose(
    #     call(ivy.layer_norm, x, norm_idxs, scale=scale, offset=offset),
    #     ivy.to_numpy(true_res),
    # )
    # # compilation test
    # if call in [helpers.torch_call]:
    #     # this is not a backend implemented function
    #     return
    # helpers.assert_compilable(ivy.layer_norm)
