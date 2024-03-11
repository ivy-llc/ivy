# global
from hypothesis import assume, strategies as st
from ivy.func_wrapper import output_to_native_arrays

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_linalg import (
    _generate_dot_dtype_and_arrays,
)
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_nn import (
    _generate_bias_data,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_nn.test_layers import (
    _lstm_helper,
)
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
)
import ivy.functional.frontends.tensorflow as tf_frontend


# --- Helpers --- #
# --------------- #


@st.composite
def _x_and_filters(
    draw,
    dtypes,
    data_format,
    padding=None,
    stride_min=1,
    stride_max=4,
    dilation_min=1,
    dilation_max=4,
    type: str = "depthwise",
):
    data_format = draw(data_format)
    dtype = draw(dtypes)
    dim = 2 if type in ["depthwise", "separable"] else 4
    if padding is None:
        padding = (st.sampled_from(["same", "valid"]),)
    padding = draw(padding)
    dilations = draw(
        st.one_of(
            st.integers(dilation_min, dilation_max),
            st.lists(
                st.integers(dilation_min, dilation_max), min_size=dim, max_size=dim
            ),
        )
    )
    fdilations = [dilations] * dim if isinstance(dilations, int) else dilations
    if type in ["depthwise", "separable"]:
        # if any value in dilations is greater than 1, tensorflow implements
        # depthwise_covn2d as an atrous depthwise convolution, in which case all values
        # in strides must be equal to 1.
        if any(x > 1 for x in fdilations):
            stride = 1
        else:
            stride = draw(st.integers(stride_min, stride_max))
    else:
        stride = draw(
            st.one_of(
                st.integers(stride_min, stride_max),
                st.lists(
                    st.integers(stride_min, stride_max), min_size=dim, max_size=dim
                ),
            )
        )
    if dim == 2:
        min_x_height = 1
        min_x_width = 1
        filter_shape = draw(
            st.tuples(
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=1, max_value=3),
                helpers.ints(min_value=1, max_value=3),
            )
        )
        min_x_height = filter_shape[0] + (filter_shape[0] - 1) * (fdilations[0] - 1)
        min_x_width = filter_shape[1] + (filter_shape[1] - 1) * (fdilations[1] - 1)
        d_in = filter_shape[2]
        if data_format == "channels_last":
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=min_x_height, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                    helpers.ints(min_value=d_in, max_value=d_in),
                )
            )
        else:
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=d_in, max_value=d_in),
                    helpers.ints(min_value=min_x_height, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                )
            )
    x = draw(
        helpers.array_values(dtype=dtype[0], shape=x_shape, min_value=0, max_value=1)
    )
    filters = draw(
        helpers.array_values(
            dtype=dtype[0], shape=filter_shape, min_value=0, max_value=1
        )
    )
    if type in ["depthwise", "separable"]:
        stride = (stride, stride)
        if isinstance(dilations, int):
            dilations = (dilations,) * dim
    return dtype, x, filters, dilations, data_format, stride, padding


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.dot",
    data=_generate_dot_dtype_and_arrays(min_num_dims=2),
)
def test_tensorflow_dot(*, data, on_device, fn_tree, frontend, test_flags, backend_fw):
    (input_dtypes, x) = data
    return helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        rtol=0.5,
        atol=0.5,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.bias_add",
    data=_generate_bias_data(keras_backend_fn=True),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_bias_add(
    *,
    data,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    data_format, dtype, x, bias = data
    helpers.test_frontend_function(
        input_dtypes=dtype * 2,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        bias=bias,
        data_format=data_format,
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.depthwise_conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["channels_last"]),
        padding=st.sampled_from(["valid", "same"]),
        type="depthwise",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_depthwise_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        depthwise_kernel=filters,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation,
    )


# mean
@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.mean",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        force_int_axis=True,
        valid_axis=True,
        min_num_dims=1,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_mean(
    *,
    dtype_x_axis,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-1,
        rtol=1e-1,
        on_device=on_device,
        x=x[0],
        axis=axis,
        keepdims=keepdims,
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.rnn",
    rnn_args=_lstm_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_rnn(
    *,
    rnn_args,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (
        input_dtypes,
        inputs,
        kernel_orig,
        recurrent_kernel_orig,
        bias_orig,
        recurrent_bias_orig,
        initial_states,
        go_backwards,
        mask,
        unroll,
        input_length,
        time_major,
        zero_output_for_mask,
        return_all_outputs,
    ) = rnn_args

    # unsupported dtype of float16 is in our _lstm_step function
    # so can't be inferred through ivy.function_unsupported_devices_and_dtypes
    assume(not (backend_fw == "torch" and input_dtypes[0] == "float16"))

    def _lstm_step(cell_inputs, cell_states):
        nonlocal kernel_orig, recurrent_kernel_orig, bias_orig, recurrent_bias_orig
        kernel = ivy.array(kernel_orig)
        recurrent_kernel = ivy.array(recurrent_kernel_orig)
        bias = ivy.array(bias_orig)
        recurrent_bias = ivy.array(recurrent_bias_orig)

        h_tm1 = cell_states[0]  # previous memory state
        c_tm1 = cell_states[1]  # previous carry state

        z = ivy.dot(cell_inputs, kernel) + bias
        z += ivy.dot(h_tm1, recurrent_kernel) + recurrent_bias

        z0, z1, z2, z3 = ivy.split(z, num_or_size_splits=4, axis=-1)

        i = ivy.sigmoid(z0)  # input
        f = ivy.sigmoid(z1)  # forget
        c = f * c_tm1 + i * ivy.tanh(z2)
        o = ivy.sigmoid(z3)  # output

        h = o * ivy.tanh(c)
        return h, [h, c]

    np_vals = [inputs, *initial_states, mask]

    if mask is None:
        np_vals.pop(-1)

    with ivy.utils.backend.ContextManager(backend_fw):
        _lstm_step_backend = outputs_to_frontend_arrays(
            inputs_to_ivy_arrays(_lstm_step)
        )
        vals = [ivy.array(val) for val in np_vals]
        if len(vals) > 3:
            inputs, init_h, init_c, mask = vals
        else:
            inputs, init_h, init_c = vals
        initial_states = [init_h, init_c]

        args = (_lstm_step_backend, inputs, initial_states)
        kwargs = {
            "go_backwards": go_backwards,
            "mask": mask,
            "constants": None,
            "unroll": unroll,
            "input_length": input_length,
            "time_major": time_major,
            "zero_output_for_mask": zero_output_for_mask,
            "return_all_outputs": return_all_outputs,
        }
        ret = tf_frontend.keras.backend.rnn(*args, **kwargs)
        ivy_ret = ivy.nested_map(lambda x: x.ivy_array, ret, shallow=False)
        ivy_idxs = ivy.nested_argwhere(ivy_ret, ivy.is_ivy_array)
        ivy_vals = ivy.multi_index_nest(ivy_ret, ivy_idxs)
        ret_np_flat = [x.to_numpy() for x in ivy_vals]

    with ivy.utils.backend.ContextManager(frontend):
        _lstm_step_gt = output_to_native_arrays(inputs_to_ivy_arrays(_lstm_step))
        import tensorflow as tf

        vals = [ivy.array(val).data for val in np_vals]
        if len(vals) > 3:
            inputs, init_h, init_c, mask = vals
        else:
            inputs, init_h, init_c = vals
        initial_states = [init_h, init_c]

        args = (_lstm_step_gt, inputs, initial_states)
        kwargs = {
            "go_backwards": go_backwards,
            "mask": mask,
            "constants": None,
            "unroll": unroll,
            "input_length": input_length,
            "time_major": time_major,
            "zero_output_for_mask": zero_output_for_mask,
            "return_all_outputs": return_all_outputs,
        }
        ret = tf.keras.backend.rnn(*args, **kwargs)
        native_idxs = ivy.nested_argwhere(ret, lambda x: isinstance(x, ivy.NativeArray))
        native_vals = ivy.multi_index_nest(ret, native_idxs)
        frontend_ret_np_flat = [x.numpy() for x in native_vals]

    helpers.value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=1e-1,
        atol=1e-1,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# sum
@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.sum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        force_int_axis=True,
        valid_axis=True,
        min_num_dims=1,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_sum(
    *,
    dtype_x_axis,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-1,
        rtol=1e-1,
        on_device=on_device,
        x=x[0],
        axis=axis,
        keepdims=keepdims,
    )
