import ivy
import numpy as np


# local
from ivy.functional.frontends.numpy import asanyarray


def test_asanyarray():
    # Test case 1: input is already an ndarray
    arr = np.array([1, 2, 3])
    assert np.array_equal(asanyarray(arr), arr)

    # Test case 2: input is not an ndarray
    ivy_arr = ivy.array([1, 2, 3])
    np_arr = np.array([1, 2, 3])
    assert np.array_equal(asanyarray(ivy_arr), np_arr)

    # Test case 3: input is not an ndarray, and dtype is specified
    ivy_arr = ivy.array([1, 2, 3], dtype=ivy.float32)
    np_arr = np.array([1, 2, 3], dtype=np.float32)
    assert np.array_equal(asanyarray(ivy_arr, dtype=np.float32), np_arr)

    # Test case 4: input is not an ndarray, and order is specified
    ivy_arr = ivy.array([[1, 2], [3, 4]])
    np_arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(asanyarray(ivy_arr, order="F"), np_arr.T)

    # Test case 5: input is not an ndarray, and keepdims is True
    ivy_arr = ivy.array([[1, 2], [3, 4]])
    np_arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(asanyarray(ivy_arr, keepdims=True), np_arr.reshape((2, 2, 1)))

    # Test case 6: input is not an ndarray, and like is specified
    ivy_arr = ivy.array([[1, 2], [3, 4]])
    np_arr = np.array([[1, 2], [3, 4]])
    like = np.zeros((2, 2))
    assert np.array_equal(asanyarray(ivy_arr, like=like), np_arr)

    # Test case 7: input is not an ndarray, and all optional parameters are specified
    ivy_arr = ivy.array([[1, 2], [3, 4]], dtype=ivy.float32)
    np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    like = np.zeros((2, 2))
    assert np.array_equal(
        asanyarray(ivy_arr, dtype=np.float32, order="F", keepdims=True, like=like),
        np_arr.T.reshape((2, 2, 1)),
    )
