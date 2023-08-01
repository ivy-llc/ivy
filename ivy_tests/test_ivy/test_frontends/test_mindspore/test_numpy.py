# global

# local


# TODO: uncomment after frontend is not required
#  to be set as backend in test_frontend_function
# @handle_frontend_test(
#     fn_tree="mindspore.numpy.array",
#     dtype_and_a=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("numeric"),
#         num_arrays=1,
#         min_num_dims=0,
#         max_num_dims=5,
#         min_dim_size=1,
#         max_dim_size=5,
#     ),
#     ndmin=st.integers(min_value=0, max_value=5),
#     copy=st.booleans(),
#     test_with_out=st.just(False),
# )
# def test_mindspore_array(
#     dtype_and_a,
#     frontend,
#     test_flags,
#     fn_tree,
#     on_device,
#     copy,
#     ndmin,
# ):
#     dtype, a = dtype_and_a
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         object=a,
#         dtype=None,
#         copy=copy,
#         ndmin=ndmin,
#     )
