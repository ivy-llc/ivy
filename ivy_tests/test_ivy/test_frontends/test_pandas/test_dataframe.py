# global
# from hypothesis import strategies as st
# import numpy as np
# local
# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_method


# CLASS_TREE = "ivy.functional.frontends.pandas.DataFrame"

# ToDo: uncomment this when frontend tests are independent of backends

# @handle_frontend_method(
#     class_tree=CLASS_TREE,
#     init_tree="pandas.DataFrame",
#     method_name="abs",
#     dtype_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"))
# )
# def test_pandas_series_abs(
#     dtype_x,
#     frontend,
#     frontend_method_data,
#     init_flags,
#     method_flags,
#     on_device,
# ):
#     # todo add castable dtypes for output
#     input_dtype, x = dtype_x
#     helpers.test_frontend_method(
#         init_input_dtypes=input_dtype,
#         init_all_as_kwargs_np={
#             "data": x[0],
#         },
#         method_input_dtypes=input_dtype,
#         method_all_as_kwargs_np={},
#         frontend_method_data=frontend_method_data,
#         init_flags=init_flags,
#         method_flags=method_flags,
#         frontend=frontend,
#         on_device=on_device,
#     )


# @handle_frontend_method(
#     class_tree=CLASS_TREE,
#     init_tree="pandas.DataFrame",
#     method_name="to_numpy",
#     dtype_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid")),
#     na_values=st.sampled_from([None, np.nan, np.inf, -np.inf]),
#     copy=st.booleans(),
# )
# def test_pandas_series_to_numpy(
#     dtype_x,
#     frontend,
#     na_values,
#     copy,
#     frontend_method_data,
#     init_flags,
#     method_flags,
#     on_device,
# ):
#     input_dtype, x = dtype_x
#     helpers.test_frontend_method(
#         init_input_dtypes=input_dtype,
#         init_all_as_kwargs_np={
#             "data": x[0],
#         },
#         method_input_dtypes=input_dtype,
#         method_all_as_kwargs_np={
#             "na_values": na_values,
#             "copy": copy,
#         },
#         frontend_method_data=frontend_method_data,
#         init_flags=init_flags,
#         method_flags=method_flags,
#         frontend=frontend,
#         on_device=on_device,
#     )
