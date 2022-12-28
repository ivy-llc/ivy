# global
from typing import Union

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions


@handle_exceptions
def is_native_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype], /) -> bool:
    """
    Determines whether the input dtype is a Native dtype.

    Parameters
    ----------
    dtype_in
        Determine whether the input data type is a native data type object.

    Returns
    -------
    ret
        Boolean, whether or not dtype_in is a native data type.

    Examples
    --------
    >>> ivy.set_backend('numpy')
    >>> ivy.is_native_dtype(np.int32)
    True

    >>> ivy.set_backend('numpy')
    >>> ivy.is_native_array(ivy.float64)
    False
    """
    try:
        return current_backend(None).is_native_dtype(dtype_in)
    except ValueError:
        return False
