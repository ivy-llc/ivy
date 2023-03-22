

# test_nonzero.py

from ivy.core.backend import get_backend
from searching import nonzero
def test_nonzero():
    # Test case 1: 1D array with all zeros
    x = ivy.zeros((5,))
    assert ivy.nonzero(x) == ()

    # Test case 2: 2D array with some non-zero elements
    x = ivy.array([[0, 1, 0], [2, 0, 3]])
    assert ivy.nonzero(x) == ((0, 1), (1, 0), (1, 2))

    # Test case 3: 3D array with size argument
    x = ivy.array([[[1, 0], [0, 3]], [[0, 0], [4, 5]]])
    assert ivy.nonzero(x, size=4, fill_value=7) == ivy.array([[0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])

    # Test case 4: 4D array with as_tuple argument
    x = ivy.array([[[[1, 0], [0, 3]], [[0, 0], [4, 5]]], [[[0, 2], [0, 0]], [[0, 0], [0, 6]]]])
    assert ivy.nonzero(x, as_tuple=False) == ivy.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]])

    # Test case 5: Zero-dimensional input
    with pytest.raises(ValueError):
        ivy.nonzero(ivy.array(3))

    # Test case 6: Container input
    with pytest.raises(NotImplementedError):
        ivy.nonzero(ivy.Container(x))
