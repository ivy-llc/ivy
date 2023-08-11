# ToDo: Uncomment this after we sort out a way of installing mindspore
# using pip and after added it to the development requirements.

# global

# local
# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test

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
