# local
from typing import Optional, Union, Sequence
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.exceptions import handle_exceptions


# dirichlet
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def dirichlet(
    alpha: Union[ivy.Array, ivy.NativeArray, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Draw size samples of dimension k from a Dirichlet distribution.
    A Dirichlet-distributed random variable can be seen as a multivariate
    generalization of a Beta distribution. The Dirichlet distribution is
    a conjugate prior of a multinomial distribution in Bayesian inference.

    Parameters
    ----------
    alpha
        Sequence of floats of length k
    size
        optional int or tuple of ints, Output shape. If the given shape is,
        e.g., (m, n), then m * n * k samples are drawn. Default is None,
        in which case a vector of length k is returned.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The drawn samples, of shape (size, k).

    Functional Examples
    -------------------

    >>> alpha = [1.0, 2.0, 3.0]
    >>> ivy.dirichlet(alpha)
    ivy.array([0.10598304, 0.21537054, 0.67864642])

    >>> alpha = [1.0, 2.0, 3.0]
    >>> ivy.dirichlet(alpha, size = (2,3))
    ivy.array([[[0.48006698, 0.07472073, 0.44521229],
        [0.55479872, 0.05426367, 0.39093761],
        [0.19531053, 0.51675832, 0.28793114]],

       [[0.12315625, 0.29823365, 0.5786101 ],
        [0.15564976, 0.50542368, 0.33892656],
        [0.1325352 , 0.44439589, 0.42306891]]])
    """
    return ivy.current_backend().dirichlet(
        alpha,
        size=size,
        dtype=dtype,
        seed=seed,
        out=out,
    )
