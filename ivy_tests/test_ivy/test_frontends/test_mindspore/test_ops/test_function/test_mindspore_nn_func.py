# global
# from hypothesis import strategies as st

# local
# TODO: uncomment after frontend is not required
#  to be set as backend in test_frontend_function

# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test

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


# @st.composite
# def _generate_bias_data(draw):
#     data_format = draw(st.sampled_from(["NC...", "N...C", None]))
#     channel_dim = 1 if data_format == "NC..." else -1
#     dtype, value, shape = draw(
#         helpers.dtype_and_values(
#             available_dtypes=helpers.get_dtypes("numeric"),
#             min_num_dims=3,
#             ret_shape=True,
#         )
#     )
#     channel_size = shape[channel_dim]
#     bias = draw(helpers.array_values(dtype=dtype[0], shape=(channel_size,)))
#     return data_format, dtype, value, bias


# @handle_frontend_test(
#     fn_tree="mindspore.ops.function.nn_func.bias_add",
#     data=_generate_bias_data(),
#     test_with_out=st.just(False),
# )
# def test_mindspore_bias_add(
#     *,
#     data,
#     frontend,
#     test_flags,
#     fn_tree,
#     backend_fw,
#     on_device,
# ):
#     data_format, dtype, value, bias = data
#     helpers.test_frontend_function(
#         input_dtypes=dtype * 2,
#         backend_to_test=backend_fw,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         value=value[0],
#         bias=bias,
#         data_format=data_format,
#     )
