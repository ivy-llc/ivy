# global
from hypothesis import strategies as st

# local
import ivy
from ivy.functional.frontends.numpy.ma.MaskedArray import MaskedArray
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _array_mask(draw):
    dtype = draw(helpers.get_dtypes("valid", prune_function=False, full=False))
    dtypes, x_mask = draw(
        helpers.dtype_and_values(
            num_arrays=2,
            dtype=[dtype[0], "bool"],
        )
    )
    return dtype[0], x_mask


# data
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_array_mask(),
)
def test_numpy_data(
    args,
):
    dtype, data = args
    x = MaskedArray(data[0], mask=data[1], dtype=dtype)
    assert ivy.all(x.data == ivy.array(data[0]))


# mask
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_array_mask(),
)
def test_numpy_mask(args):
    dtype, data = args
    x = MaskedArray(data[0], mask=ivy.array(data[1]), dtype=dtype, shrink=False)
    assert ivy.all(x.mask == ivy.array(data[1]))


# fill_value
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    dtype_x_mask=_array_mask(),
    fill=st.integers(),
)
def test_numpy_fill_value(
    dtype_x_mask,
    fill,
):
    dtype, data = dtype_x_mask
    x = MaskedArray(data[0], mask=data[1], dtype=dtype, fill_value=fill)
    assert x.fill_value == ivy.array(fill, dtype=dtype)


# hardmask
@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    dtype_x_mask=_array_mask(),
    hard=st.booleans(),
)
def test_numpy_hardmask(dtype_x_mask, hard):
    dtype, data = dtype_x_mask
    x = MaskedArray(data[0], mask=data[1], dtype=dtype, hard_mask=hard)
    assert x.hardmask == hard


# dtype
@handle_frontend_test(fn_tree="numpy.add", dtype_x_mask=_array_mask())  # dummy fn_tree
def test_numpy_dtype(dtype_x_mask):
    dtype, data = dtype_x_mask
    x = MaskedArray(data[0], mask=data[1], dtype=dtype)
    assert x.dtype == dtype


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
# def test_numpy___getitem__(
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
# def test_numpy___setitem__(
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
