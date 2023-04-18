import ivy
import ivy.functional.frontends.numpy as np_frontend
import numpy as np
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def ifft(a, n=None, axis=-1, norm=None):
    a = ivy.array(a, dtype=ivy.complex128)
    if norm is None:
        norm = "backward"
    return ivy.ifft(a, axis, norm=norm, n=n)


@to_ivy_arrays_and_back
def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.
    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.
    Returns
    -------
    y : ndarray
        The shifted array.
    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.
    Examples
    --------
    >>> import ivy.functional.frontends.numpy as np_frontend
    >>> arr = np_frontend.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> arr
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9]])
    >>> np_frontend.ifftshift(arr, axes=(0, 1))
    array([[ 5,  6,  4],
           [ 9,  7,  8],
           [ 3,  1,  2]])
    """
    x = np_frontend.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, (int,
                           type(ivy.uint8),
                           type(ivy.uint16),
                           type(ivy.uint32),
                           type(ivy.uint64))):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]
        
    # Change to ivy `numpy.core.roll()` equivalent when available
    return np.roll(x, shift, axes)
    
