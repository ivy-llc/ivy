# ToDo: Uncomment this after we sort out a way of installing mindspore
# using pip and after added it to the development requirements.

# global

# local
# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test
# from hypothesis import strategies as st
# from ivy_tests.test_ivy.test_functional.test_nn.test_layers import (
#     _assume_tf_dilation_gt_1,
# )
# import ivy


# # softsign
# @handle_frontend_test(
#     fn_tree="mindspore.ops.softsign",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes(kind="float", full=False, key="dtype"),
#         safety_factor_scale="log",
#         small_abs_safety_factor=20,
#     ),
# )
# def test_mindspore_softsign(
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


# @st.composite
# def x_and_filters(draw, dim: int = 2):
#     if not isinstance(dim, int):
#         dim = draw(dim)
#     strides = draw(
#         st.one_of(
#             st.lists(st.integers(min_value=1, max_value=3), min_size=dim, max_size=dim),
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
#             st.lists(st.integers(min_value=1, max_value=3), min_size=dim, max_size=dim),
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
#     fn_tree="mindspore.ops.Conv1d",
#     dtype_vals=x_and_filters(dim=1),
# )
# def test_torch_conv1d(
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
#     fn_tree="mindspore.ops.Conv2d",
#     dtype_vals=x_and_filters(dim=2),
# )
# def test_torch_conv2d(
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
#     fn_tree="mindspore.ops.Conv3d",
#     dtype_vals=x_and_filters(dim=3),
# )
# def test_torch_conv3d(
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
