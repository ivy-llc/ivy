import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
import functools
import operator
import warnings

from numpy.core import (
    array, asarray, zeros, empty, empty_like, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, all, Inf, dot,
    add, multiply, sqrt, fastCopyAndTranspose, sum, isfinite,
    finfo, errstate, geterrobj, moveaxis, amin, amax, product, abs,
    atleast_2d, intp, asanyarray, object_, matmul,
    swapaxes, divide, count_nonzero, isnan, sign, argsort, sort,
    reciprocal
)
from numpy.core.multiarray import normalize_axis_index
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import _umath_linalg

@array_function_dispatch(_solve_dispatcher)
def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.
    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.
    Parameters
    ----------
    a : (..., M, M) array_like
        Coefficient matrix.
    b : {(..., M,), (..., M, K)}, array_like
        Ordinate or "dependent variable" values.
    Returns
    -------
    x : {(..., M,), (..., M, K)} ndarray
        Solution to the system a x = b.  Returned shape is identical to `b`.
    Raises
    ------
    LinAlgError
        If `a` is singular or not square.
    See Also
    --------
    scipy.linalg.solve : Similar function in SciPy.
    Notes
    -----
    .. versionadded:: 1.8.0
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    The solutions are computed using LAPACK routine ``_gesv``.
    `a` must be square and of full-rank, i.e., all rows (or, equivalently,
    columns) must be linearly independent; if either is not true, use
    `lstsq` for the least-squares best "solution" of the
    system/equation.
    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pg. 22.
    Examples
    --------
    Solve the system of equations ``x0 + 2 * x1 = 1`` and ``3 * x0 + 5 * x1 = 2``:
    >>> a = np.array([[1, 2], [3, 5]])
    >>> b = np.array([1, 2])
    >>> x = np.linalg.solve(a, b)
    >>> x
    array([-1.,  1.])
    Check that the solution is correct:
    >>> np.allclose(np.dot(a, x), b)
    True
    """
    a, _ = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    b, wrap = _makearray(b)
    t, result_t = _commonType(a, b)

    # We use the b = (..., M,) logic, only if the number of extra dimensions
    # match exactly
    if b.ndim == a.ndim - 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve

    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    r = gufunc(a, b, signature=signature, extobj=extobj)

    return wrap(r.astype(result_t, copy=False))



@to_ivy_arrays_and_back
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)


@to_ivy_arrays_and_back
def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


@to_ivy_arrays_and_back
def eigh(x, /, *, lower=True, symmetrize_input=True, sort_eigenvalues=True):
    UPLO = "L" if lower else "U"

    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.eigh(x, UPLO=UPLO)



    
