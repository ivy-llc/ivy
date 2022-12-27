# # global
# from hypothesis import strategies as st
# import numpy as np

# # local
# import ivy
# from ivy.functional.frontends.numpy.ma.MaskedArray import MaskedArray
# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test
# import ivy.functional.backends.torch as ivy_torch


# @st.composite
# def _getitem_helper(draw):
#     dtype_x_index = draw(
#         helpers.array_indices_axis(
#             array_dtypes=helpers.get_dtypes("numeric"),
#             indices_dtypes=ivy_torch.valid_int_dtypes,
#             indices_same_dims=True,
#         )
#     )
#     dtype, x, index = dtype_x_index[:3]
#     mask = draw(
#         helpers.dtype_and_values(
#             available_dtypes=helpers.get_dtypes("bool"),
#             shape=x.shape,
#         )
#     )
#     return dtype[0], x, mask[1][0], index


# # __getitem__
# @handle_frontend_test(
#     fn_tree="numpy.add",  # dummy fn_tree
#     args=_getitem_helper(),
# )
# def test_numpy_maskedarray_special_getitem(
#     args,
# ):
#     dtype, x, mask, index = args
#     data = MaskedArray(x, mask=mask, dtype=dtype)
#     ret = data.__getitem__(index)
#     data_gt = np.ma.MaskedArray(x, mask=mask, dtype=dtype)
#     ret_gt = data_gt.__getitem__(index)
#     ret = ivy.to_numpy(ivy.flatten(ret.data))
#     ret_gt = np.array(np.ravel(ret_gt))
#     helpers.value_test(
#         ret_np_flat=ret,
#         ret_np_from_gt_flat=ret_gt,
#         ground_truth_backend="numpy",
#     )


# @st.composite
# def _setitem_helper(draw):
#     dtype_x_index = draw(
#         helpers.array_indices_axis(
#             array_dtypes=st.shared(helpers.get_dtypes("numeric"), key="dtype"),
#             indices_dtypes=ivy_torch.valid_int_dtypes,
#             indices_same_dims=True,
#         )
#     )
#     dtype, x, index = dtype_x_index[:3]
#     mask = draw(
#         helpers.dtype_and_values(
#             available_dtypes=helpers.get_dtypes("bool"),
#             shape=x.shape,
#         )
#     )
#     value = draw(
#         helpers.dtype_and_values(
#             available_dtypes=st.shared(helpers.get_dtypes("numeric"), key="dtype"),
#             shape=index.shape,
#         )
#     )
#     return dtype[0], x, mask[1][0], index, value[1][0]


# # __setitem__
# @handle_frontend_test(
#     fn_tree="numpy.add",  # dummy fn_tree
#     args=_setitem_helper(),
# )
# def test_numpy_maskedarray_special_setitem(
#     args,
# ):
#     dtype, x, mask, index, value = args
#     data = MaskedArray(x, mask=mask, dtype=dtype)
#     data_gt = np.ma.MaskedArray(x, mask=mask, dtype=dtype)
#     data = data.__setitem__(index, value)
#     data_gt.__setitem__(index, value)
#     ret = ivy.to_numpy(ivy.flatten(data.data))
#     ret_gt = np.array(np.ravel(data_gt))
#     helpers.value_test(
#         ret_np_flat=ret,
#         ret_np_from_gt_flat=ret_gt,
#         ground_truth_backend="numpy",
#     )
