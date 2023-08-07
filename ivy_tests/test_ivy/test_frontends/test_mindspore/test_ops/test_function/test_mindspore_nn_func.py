# global
# from hypothesis import strategies as st

# local
# TODO: uncomment after frontend is not required
#  to be set as backend in test_frontend_function

# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test
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
#     pad_width=st.lists(st.tuples(st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5))),
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
