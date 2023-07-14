# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy
from ivy.functional.frontends.numpy import broadcast
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _broadcastable_arrays(draw):
    num_of_array = draw(st.integers(1, 3))
    shapes = draw(helpers.mutually_broadcastable_shapes(num_shapes=num_of_array))
    xs = []
    for i in range(num_of_array):
        xs.append(
            draw(
                helpers.array_values(dtype=helpers.get_dtypes("valid"), shape=shapes[i])
            )
        )
    return xs


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_shape(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    assert ret.shape == ret_gt.shape


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_size(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    assert ret.size == ret_gt.size


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_nd(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    assert ret.nd == ret_gt.nd


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_ndim(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    assert ret.ndim == ret_gt.ndim


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_numiter(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    assert ret.numiter == ret_gt.numiter


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_index(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    assert ret.index == ret_gt.index
    for _ in zip(ret, ret_gt):
        assert ret.index == ret_gt.index


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_property_iters(args):
    ret = list(map(list, broadcast(*args).iters))
    ret_gt = np.array(list(map(list, np.broadcast(*args).iters)))
    assert ivy.all(ret == ret_gt)


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    args=_broadcastable_arrays(),
)
def test_numpy_broadcast_method_reset(args):
    ret = broadcast(*args)
    ret_gt = np.broadcast(*args)
    for _ in zip(ret, ret_gt):
        pass
    ret.reset()
    ret_gt.reset()
    assert ret.index == ret_gt.index
