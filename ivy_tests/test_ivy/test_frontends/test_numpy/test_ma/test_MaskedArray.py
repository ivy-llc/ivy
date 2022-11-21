# global
from hypothesis import given, strategies as st

# local
import numpy as np
from ivy.functional.frontends.numpy.ma.MaskedArray import MaskedArray
import ivy_tests.test_ivy.helpers as helpers


@st.composite
def _getitem_helper(draw):
    arr_size = draw(helpers.ints(min_value=2, max_value=10))
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=(arr_size,)
        )
    )
    mask = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("bool"),
            shape=(arr_size,)
        )
    )
    index = draw(helpers.ints(min_value=0, max_value=arr_size - 1))
    return dtype, x, mask, index


# __getitem__
@given(
    args=_getitem_helper(),
)
def test_numpy_maskedarray_special_getitem(
    args,
):
    dtype, x, mask, index = args
    data = MaskedArray(x[0], mask=mask, dtype=dtype[0])
    ret = data.__getitem__(index)
    data_gt = np.ma.MaskedArray(x[0], mask=mask, dtype=dtype[0])
    ret_gt = data_gt.at[index].get()
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


@st.composite
def _setitem_helper(draw):
    arr_size = draw(helpers.ints(min_value=2, max_value=10))
    available_dtypes = draw(helpers.get_dtypes("float")) + draw(helpers.get_dtypes("int"))
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=st.sampled_from(available_dtypes),
            shape=(arr_size,)
        )
    )
    mask = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("bool"),
            shape=(arr_size,)
        )
    )
    index = draw(helpers.ints(min_value=0, max_value=arr_size - 1))
    if 'float' in dtype[0]:
        value = draw(helpers.floats())[0]
    else:
        value = draw(helpers.ints())[0]
    return dtype, x, mask, index, value


# __setitem__
@given(
    args=_setitem_helper(),
)
def test_numpy_maskedarray_special_setitem(
    args,
):
    dtype, x, mask, index, value = args
    data = MaskedArray(x[0], mask=mask, dtype=dtype[0])
    ret = data.__setitem__(index, value)
    data_gt = np.ma.MaskedArray(x[0], mask=mask, dtype=dtype[0])
    ret_gt = data_gt.at[index].set(value)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )
