"""Collection of tests for sorting functions."""

# global
# import numpy as np
# from hypothesis import given, strategies as st

# local
# import ivy_tests.test_ivy.helpers as helpers
# import ivy.functional.backends.numpy as ivy_np


# argsort
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtypes),
#     as_variable=st.booleans(),
#     with_out=st.booleans(),
#     num_positional_args=st.integers(0, 1),
#     native_array=st.booleans(),
#     container=st.booleans(),
#     instance_method=st.booleans(),
# )
# def test_argsort(
#     dtype_and_x,
#     as_variable,
#     with_out,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     dtype, x = dtype_and_x
#     # smoke this for torch
#     if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
#         return
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         with_out,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "argsort",
#         x=np.asarray(x, dtype=dtype),
#     )


# sort
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtypes),
#     as_variable=st.booleans(),
#     with_out=st.booleans(),
#     num_positional_args=st.integers(0, 1),
#     native_array=st.booleans(),
#     container=st.booleans(),
#     instance_method=st.booleans(),
# )
# def test_sort(
#     dtype_and_x,
#     as_variable,
#     with_out,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     dtype, x = dtype_and_x
#     # smoke this for torch
#     if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
#         return
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         with_out,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "sort",
#         x=np.asarray(x, dtype=dtype),
#     )
