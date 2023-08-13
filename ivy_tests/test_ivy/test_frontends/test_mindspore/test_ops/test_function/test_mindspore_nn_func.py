# global
# from hypothesis import assume, strategies as st

# local
# TODO: uncomment after frontend is not required
#  to be set as backend in test_frontend_function

# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test
# from ivy_tests.test_ivy.test_functional.test_nn.test_layers import (
#     _assume_tf_dilation_gt_1,
# )
# import ivy

# import math
#
# #dropout2d
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.dropout2d",
#     d_type_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=1,
#         shared_dtype=True,
#         min_value=2,
#         max_value=5,
#         min_dim_size=4,
#         shape=(
#             st.integers(min_value=2, max_value=10),
#             4,
#             st.integers(min_value=12, max_value=64),
#             st.integers(min_value=12, max_value=64),
#         ),
#     ),
#     p=st.floats(min_value=0.0, max_value=1.0),
#     training=st.booleans(),
# )
# def test_mindspore_dropout2d(
#     *,
#     d_type_and_x,
#     p,
#     training,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     dtype, x = d_type_and_x
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#         p=p,
#         training=training,
#     )


# selu
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.selu",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         safety_factor_scale="log",
#         small_abs_safety_factor=20,
#     ),
# )
# def test_mindspore_selu(
#     *,
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         x=x[0],
#     )

# kl_div
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.kl_div",
#     p=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=1,
#         shared_dtype=True,
#         min_value=2,
#         max_value=5,
#         min_dim_size=4,
#     ),
#     q=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=1,
#         shared_dtype=True,
#         min_value=2,
#         max_value=5,
#         min_dim_size=4,
#     ),
#     reduction=st.sampled_from(["none", "sum", "mean"]),
# )
# def test_mindspore_kl_div(
#     *,
#     p,
#     q,
#     reduction,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     helpers.test_frontend_function(
#         input_dtypes=p[0],
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         p=p[1],
#         q=q[1],
#         reduction=reduction,
#     )

# dropout3d
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.dropout3d",
#     d_type_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=1,
#         shared_dtype=True,
#         min_value=2,
#         max_value=5,
#         min_dim_size=5,
#         shape=(
#             st.integers(min_value=2, max_value=10),
#             st.integers(min_value=12, max_value=64),
#             st.integers(min_value=12, max_value=64),
#             st.integers(min_value=12, max_value=64),
#         ),
#     ),
#     p=st.floats(min_value=0.0, max_value=1.0),
#     training=st.booleans(),
# )
# def test_mindspore_dropout3d(
#     *,
#     d_type_and_x,
#     p,
#     training,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     dtype, x = d_type_and_x
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#         p=p,
#         training=training,
#     )


# def _size_strategy():
#     return st.one_of(
#         st.integers(min_value=1, max_value=10),
#         st.tuples(st.integers(min_value=1, max_value=10)),
#         st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3),
#     )

# def _scale_factor_strategy():
#     return st.one_of(
#         st.floats(min_value=0.1, max_value=2.0),
#         st.tuples(st.floats(min_value=0.1, max_value=2.0)),
#         st.lists(st.floats(min_value=0.1, max_value=2.0), min_size=3, max_size=3),
#     )

# def _size_and_scale_factor_strategy():
#     return st.one_of(
#         st.tuples(size_strategy(), st.just(None)),
#         st.tuples(st.just(None), scale_factor_strategy()),
#         st.tuples(size_strategy(), scale_factor_strategy()),
#     )


# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.interpolate",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=1,
#         shared_dtype=True,
#     ),
#     mode=st.sampled_from(
#         [
#             "nearest",
#             "linear",
#             "bilinear",
#             "bicubic",
#             "trilinear",
#             "area",
#             "nearest-exact",
#         ]
#     ),
#     align_corners=st.booleans(),
#     recompute_scale_factor=st.booleans(),
#     size_and_scale_factor = _size_and_scale_factor_strategy()
# )
# def test_mindspore_interpolate(
#     *,
#     dtype_and_x,
#     size,
#     scale_factor,
#     mode,
#     align_corners,
#     recompute_scale_factor,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     dtype, x = dtype_and_x
#     size,scale_factor = size_and_scale_factor


#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#         size=size,
#         scale_factor=scale_factor,
#         mode=mode,
#         align_corners=align_corners,
#         recompute_scale_factor=recompute_scale_factor,
#     )


# pad
# @handle_frontend_test(
#     fn_tree="pad",
#     input=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=1,
#         shared_dtype=True,
#         min_value=2,
#         max_value=5,
#         min_dim_size=4,
#     ),
#     pad_width=st.lists(st.tuples(st.integers(min_value=0, max_value=5),
#                                  st.integers(min_value=0, max_value=5))),
#     mode=st.sampled_from(['constant', 'reflect', 'replicate', 'circular']),
#     constant_values=st.floats(min_value=0.0, max_value=1.0),
# )
# def test_mindspore_pad(
#     *,
#     input,
#     pad_width,
#     mode,
#     constant_values,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     helpers.test_frontend_function(
#         input_dtypes=input[0],
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=input[1],
#         pad_width=pad_width,
#         mode=mode,
#         constant_values=constant_values,
#     )


# adaptive_avg_pool2d
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.adaptive_avg_pool2d",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("float"),
#         min_num_dims=4,
#         max_num_dims=4,
#         min_dim_size=1,
#         max_value=100,
#         min_value=-100,
#     ),
#     output_size=st.one_of(
#         st.tuples(
#             helpers.ints(min_value=1, max_value=5),
#             helpers.ints(min_value=1, max_value=5),
#         ),
#         helpers.ints(min_value=1, max_value=5),
#     ),
# )
# def test_mindspore_adaptive_avg_pool2d(
#     *,
#     dtype_and_x,
#     output_size,
#     test_flags,
#     frontend,
#     on_device,
#     fn_tree,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         on_device=on_device,
#         fn_tree=fn_tree,
#         x=x[0],
#         output_size=output_size,
#     )


# log_softmax
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.log_softmax",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         safety_factor_scale="log",
#         small_abs_safety_factor=20,
#     ),
# )
# def test_mindspore_log_softmax(
#     *,
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x

# def _is_same_padding(padding, stride, kernel_size, input_shape):
#     output_shape = tuple(
#         [
#             (input_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
#             for i in range(len(padding))
#         ]
#     )
#     return all(
#         [
#             output_shape[i] == math.ceil(input_shape[i] / stride[i])
#             for i in range(len(padding))
#         ]
#     )


# def _calculate_same_padding(kernel_size, stride, shape):
#     padding = tuple(
#         [
#             max(
#                 0,
#                 math.ceil(((shape[i] - 1) * stride[i] +
#                            kernel_size[i] - shape[i]) / 2),
#             )
#             for i in range(len(kernel_size))
#         ]
#     )
#     if all([kernel_size[i] / 2 >= padding[i] for i in range(len(kernel_size))]):
#         if _is_same_padding(padding, stride, kernel_size, shape):
#             return padding
#     return (0, 0)


# # avg_pool2d
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.avg_pool2d",
#     dtype_x_k_s=helpers.arrays_for_pooling(
#         min_dims=4,
#         max_dims=4,
#         min_side=1,
#         max_side=4,
#     ),
#     pad_mode=st.booleans(),
#     count_include_pad=st.booleans(),
#     test_with_out=st.just(False),
# )
# def test_torch_avg_pool2d(
#     dtype_x_k_s,
#     count_include_pad,
#     pad_mode,
#     *,
#     test_flags,
#     frontend,
#     backend_fw,
#     fn_tree,
#     on_device,
# ):
#     input_dtype, x, kernel_size, stride, pad_name = dtype_x_k_s

#     if len(stride) == 1:
#         stride = (stride[0], stride[0])

#     if pad_name == "SAME":
#         padding = _calculate_same_padding(kernel_size, stride, x[0].shape[2:])
#     else:
#         padding = (0, 0)

#     x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], *x[0].shape[1:-1]))

#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         backend_to_test=backend_fw,
#         test_flags=test_flags,
#         frontend=frontend,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         pad_mode=pad_mode,
#         count_include_pad=count_include_pad,
#         divisor_override=None,
#     )


# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.smooth_l1_loss",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("float"),
#         num_arrays=2,
#         allow_inf=False,
#         shared_dtype=True,
#     ),
#     size_average=st.booleans(),
#     reduce=st.booleans(),
#     beta=st.sampled_from([1.0, 0.5, 0.1]),
#     test_with_out=st.just(False),
# )
# def test_mindspore_smooth_l1_loss(
#     *,
#     dtype_and_x,
#     size_average,
#     reduce,
#     beta,
#     frontend,
#     test_flags,
#     fn_tree,
#     backend_fw,
#     on_device,
# ):
#     input_dtype, x = dtype_and_x
#     pred_dtype, pred = input_dtype[0], x[0]
#     true_dtype, true = input_dtype[1], x[1]
#     helpers.test_frontend_function(
#         input_dtypes=[pred_dtype, true_dtype],
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=pred,
#         target=true,
#         size_average=size_average,
#         reduce=reduce,
#         beta=beta,
#     )


# # softshrink
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.softshrink",
#     dtype_and_input=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#     ),
#     lambd=helpers.floats(min_value=0, max_value=1, exclude_min=True),
# )
# def test_mindspore_softshrink(
#     *,
#     dtype_and_input,
#     lambd,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
#     backend_fw,
# ):
#     input_dtype, x = dtype_and_input
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         x=x[0],
#         lambd=lambd,
#     )


# gumbel_softmax
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.gumbel_softmax",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#     ),
#     tau=st.floats(min_value=0),
#     hard=st.booleans(),
#     dim=st.integers(),
#     test_with_out=st.just(False),
#     test_inplace=st.booleans(),
# )
# def test_torch_gumbel_softmax(
#     *,
#     dtype_and_x,
#     tau,
#     hard,
#     dim,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
#     backend_fw,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         test_values=False,
#         logits=x[0],
#         tau=tau,
#         hard=hard,
#         dim=dim,
#     )


# FastGelu
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.fast_gelu",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("float"),
#     ),
# )
# def test_mindspore_fast_gelu(
#     dtype_and_x,
#     *,
#     test_flags,
#     frontend,
#     on_device,
#     fn_tree,
# ):
#     input_dtype, x = dtype_and_x


#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,

#         x=x[0],

#         input=x[0],

#     )


# # flatten
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.flatten",
#     dtype_input_axes=helpers.dtype_values_axis(
#         available_dtypes=helpers.get_dtypes("valid"),
#         valid_axis=True,
#         min_num_dims=1,
#         min_axes_size=2,
#         max_axes_size=2,
#     ),
# )
# def test_mindspore_flatten(
#     *,
#     dtype_input_axes,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
#     backend_fw,
# ):
#     dtype, input, axes = dtype_input_axes
#     if isinstance(axes, int):
#         start_dim = axes
#         end_dim = -1
#     else:
#         start_dim = axes[0]
#         end_dim = axes[1]
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=input[0],
#         order='C',
#         start_dim=start_dim,
#         end_dim=end_dim,
#     )


# @st.composite
# def x_and_filters(draw, dim: int = 2):
#     if not isinstance(dim, int):
#         dim = draw(dim)
#     strides = draw(
#         st.one_of(
#             st.lists(
#             st.integers(min_value=1, max_value=3), min_size=dim, max_size=dim,
#             ),
#             st.integers(min_value=1, max_value=3),
#         )
#     )
#
#     pad_mode = draw(st.sampled_from(["valid", "same", "pad"]))
#
#     padding = draw(
#         st.one_of(
#             st.integers(min_value=1, max_value=3),
#             st.lists(
#                 st.integers(min_value=1, max_value=2), min_size=dim, max_size=dim
#             ),
#         )
#     )
#
#     batch_size = draw(st.integers(1, 5))
#     filter_shape = draw(
#         helpers.get_shape(
#             min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
#         )
#     )
#     dtype = draw(helpers.get_dtypes("float", full=False))
#     input_channels = draw(st.integers(1, 3))
#     output_channels = draw(st.integers(1, 3))
#     group_list = [i for i in range(1, 3)]
#
#     group_list = list(filter(lambda x: (input_channels % x == 0), group_list))
#
#     fc = draw(st.sampled_from(group_list))
#     dilations = draw(
#         st.one_of(
#             st.lists(
#                 st.integers(min_value=1, max_value=3), min_size=dim, max_size=dim,
#             ),
#             st.integers(min_value=1, max_value=3),
#         )
#     )
#     full_dilations = [dilations] * dim if isinstance(dilations, int) else dilations
#
#     x_dim = []
#     for i in range(dim):
#         min_x = filter_shape[i] + (filter_shape[i] - 1) * (full_dilations[i] - 1)
#         x_dim.append(draw(st.integers(min_x, 15)))
#     x_dim = tuple(x_dim)
#
#     output_channels = output_channels * fc
#     filter_shape = (output_channels, input_channels // fc) + filter_shape
#
#     x_shape = (batch_size, input_channels) + x_dim
#     vals = draw(
#         helpers.array_values(
#             dtype=dtype[0],
#             shape=x_shape,
#             min_value=0.0,
#             max_value=1.0,
#         )
#     )
#     filters = draw(
#         helpers.array_values(
#             dtype=dtype[0],
#             shape=filter_shape,
#             min_value=0.0,
#             max_value=1.0,
#         )
#     )
#     bias = draw(
#         helpers.array_values(
#             dtype=dtype[0],
#             shape=(output_channels,),
#             min_value=0.0,
#             max_value=1.0,
#         )
#     )
#
#     return dtype, vals, filters, bias, dilations, strides, padding, fc, pad_mode
#
#
# # conv1d
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.Conv1d",
#     dtype_vals=x_and_filters(dim=1),
# )
# def test_mindspore_conv1d(
#     *,
#     dtype_vals,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
#     backend_fw,
# ):
#     dtype, vals, weight, bias, dilations, strides, padding, fc, pad_mode = dtype_vals
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=vals,
#         weight=weight,
#         bias=bias,
#         stride=strides,
#         padding=padding,
#         dilation=dilations,
#         groups=fc,
#         pad_mode=pad_mode,
#     )
#
#
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.Conv2d",
#     dtype_vals=x_and_filters(dim=2),
# )
# def test_mindspore_conv2d(
#     *,
#     dtype_vals,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
#     backend_fw,
# ):
#     dtype, vals, weight, bias, dilations, strides, padding, fc, pad_mode = dtype_vals
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=vals,
#         weight=weight,
#         bias=bias,
#         stride=strides,
#         padding=padding,
#         dilation=dilations,
#         groups=fc,
#         pad_mode=pad_mode,
#     )
#
#
# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.Conv3d",
#     dtype_vals=x_and_filters(dim=3),
# )
# def test_mindspore_conv3d(
#     *,
#     dtype_vals,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
#     backend_fw,
# ):
#     dtype, vals, weight, bias, dilations, strides, padding, fc, pad_mode = dtype_vals
#     # ToDo: Enable gradient tests for dilations > 1 when tensorflow supports it.
#     _assume_tf_dilation_gt_1(ivy.current_backend_str(), on_device, dilations)
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=vals,
#         weight=weight,
#         bias=bias,
#         stride=strides,
#         padding=padding,
#         dilation=dilations,
#         groups=fc,
#         pad_mode=pad_mode,
#     )
