"""
Tests for linalg functions

https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html

and

https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html

Note: this file currently mixes both the required linear algebra functions and
functions from the linalg extension. The functions in the latter are not
required, but we don't yet have a clean way to disable only those tests (see https://github.com/data-apis/array-api-tests/issues/25).

"""

import pytest
from hypothesis import assume, given
from hypothesis.strategies import (booleans, composite, none, tuples, integers,
                                   shared, sampled_from, one_of, data, just)
from ndindex import iter_indices

from .array_helpers import assert_exactly_equal, asarray
from .hypothesis_helpers import (xps, dtypes, shapes, kwargs, matrix_shapes,
                                 square_matrix_shapes, symmetric_matrices,
                                 positive_definite_matrices, MAX_ARRAY_SIZE,
                                 invertible_matrices, two_mutual_arrays,
                                 mutually_promotable_dtypes, one_d_shapes,
                                 two_mutually_broadcastable_shapes,
                                 SQRT_MAX_ARRAY_SIZE, finite_matrices,
                                 rtol_shared_matrix_shapes, rtols)
from . import dtype_helpers as dh
from . import pytest_helpers as ph
from . import shape_helpers as sh

from . import _array_module
from . import _array_module as xp
from ._array_module import linalg

pytestmark = pytest.mark.ci

# Standin strategy for not yet implemented tests
todo = none()

def _test_stacks(f, *args, res=None, dims=2, true_val=None,
                 matrix_axes=(-2, -1),
                 assert_equal=assert_exactly_equal, **kw):
    """
    Test that f(*args, **kw) maps across stacks of matrices

    dims is the number of dimensions f(*args, *kw) should have for a single n
    x m matrix stack.

    matrix_axes are the axes along which matrices (or vectors) are stacked in
    the input.

    true_val may be a function such that true_val(*x_stacks, **kw) gives the
    true value for f on a stack.

    res should be the result of f(*args, **kw). It is computed if not passed
    in.

    """
    if res is None:
        res = f(*args, **kw)

    shapes = [x.shape for x in args]

    # Assume the result is stacked along the last 'dims' axes of matrix_axes.
    # This holds for all the functions tested in this file
    res_axes = matrix_axes[::-1][:dims]

    for (x_idxes, (res_idx,)) in zip(
            iter_indices(*shapes, skip_axes=matrix_axes),
            iter_indices(res.shape, skip_axes=res_axes)):
        x_idxes = [x_idx.raw for x_idx in x_idxes]
        res_idx = res_idx.raw

        res_stack = res[res_idx]
        x_stacks = [x[x_idx] for x, x_idx in zip(args, x_idxes)]
        decomp_res_stack = f(*x_stacks, **kw)
        assert_equal(res_stack, decomp_res_stack)
        if true_val:
            assert_equal(decomp_res_stack, true_val(*x_stacks))

def _test_namedtuple(res, fields, func_name):
    """
    Test that res is a namedtuple with the correct fields.
    """
    # isinstance(namedtuple) doesn't work, and it could be either
    # collections.namedtuple or typing.NamedTuple. So we just check that it is
    # a tuple subclass with the right fields in the right order.

    assert isinstance(res, tuple), f"{func_name}() did not return a tuple"
    assert len(res) == len(fields), f"{func_name}() result tuple not the correct length (should have {len(fields)} elements)"
    for i, field in enumerate(fields):
        assert hasattr(res, field), f"{func_name}() result namedtuple doesn't have the '{field}' field"
        assert res[i] is getattr(res, field), f"{func_name}() result namedtuple '{field}' field is not in position {i}"

@pytest.mark.xp_extension('linalg')
@given(
    x=positive_definite_matrices(),
    kw=kwargs(upper=booleans())
)
def test_cholesky(x, kw):
    res = linalg.cholesky(x, **kw)

    assert res.shape == x.shape, "cholesky() did not return the correct shape"
    assert res.dtype == x.dtype, "cholesky() did not return the correct dtype"

    _test_stacks(linalg.cholesky, x, **kw, res=res)

    # Test that the result is upper or lower triangular
    if kw.get('upper', False):
        assert_exactly_equal(res, _array_module.triu(res))
    else:
        assert_exactly_equal(res, _array_module.tril(res))


@composite
def cross_args(draw, dtype_objects=dh.numeric_dtypes):
    """
    cross() requires two arrays with a size 3 in the 'axis' dimension

    To do this, we generate a shape and an axis but change the shape to be 3
    in the drawn axis.

    """
    shape = list(draw(shapes()))
    size = len(shape)
    assume(size > 0)

    kw = draw(kwargs(axis=integers(-size, size-1)))
    axis = kw.get('axis', -1)
    shape[axis] = 3
    shape = tuple(shape)

    mutual_dtypes = shared(mutually_promotable_dtypes(dtypes=dtype_objects))
    arrays1 = xps.arrays(
        dtype=mutual_dtypes.map(lambda pair: pair[0]),
        shape=shape,
    )
    arrays2 = xps.arrays(
        dtype=mutual_dtypes.map(lambda pair: pair[1]),
        shape=shape,
    )
    return draw(arrays1), draw(arrays2), kw

@pytest.mark.xp_extension('linalg')
@given(
    cross_args()
)
def test_cross(x1_x2_kw):
    x1, x2, kw = x1_x2_kw

    axis = kw.get('axis', -1)
    err = "test_cross produced invalid input. This indicates a bug in the test suite."
    assert x1.shape == x2.shape, err
    shape = x1.shape
    assert x1.shape[axis] == x2.shape[axis] == 3, err

    res = linalg.cross(x1, x2, **kw)

    assert res.dtype == dh.result_type(x1.dtype, x2.dtype), "cross() did not return the correct dtype"
    assert res.shape == shape, "cross() did not return the correct shape"

    def exact_cross(a, b):
        assert a.shape == b.shape == (3,), "Invalid cross() stack shapes. This indicates a bug in the test suite."
        return asarray([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        ], dtype=res.dtype)

    # We don't want to pass in **kw here because that would pass axis to
    # cross() on a single stack, but the axis is not meaningful on unstacked
    # vectors.
    _test_stacks(linalg.cross, x1, x2, dims=1, matrix_axes=(axis,), res=res, true_val=exact_cross)

@pytest.mark.xp_extension('linalg')
@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=square_matrix_shapes),
)
def test_det(x):
    res = linalg.det(x)

    assert res.dtype == x.dtype, "det() did not return the correct dtype"
    assert res.shape == x.shape[:-2], "det() did not return the correct shape"

    _test_stacks(linalg.det, x, res=res, dims=0)

    # TODO: Test that res actually corresponds to the determinant of x

@pytest.mark.xp_extension('linalg')
@given(
    x=xps.arrays(dtype=dtypes, shape=matrix_shapes()),
    # offset may produce an overflow if it is too large. Supporting offsets
    # that are way larger than the array shape isn't very important.
    kw=kwargs(offset=integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE))
)
def test_diagonal(x, kw):
    res = linalg.diagonal(x, **kw)

    assert res.dtype == x.dtype, "diagonal() returned the wrong dtype"

    n, m = x.shape[-2:]
    offset = kw.get('offset', 0)
    # Note: the spec does not specify that offset must be within the bounds of
    # the matrix. A large offset should just produce a size 0 in the last
    # dimension.
    if offset < 0:
        diag_size = min(n, m, max(n + offset, 0))
    elif offset == 0:
        diag_size = min(n, m)
    else:
        diag_size = min(n, m, max(m - offset, 0))

    assert res.shape == (*x.shape[:-2], diag_size), "diagonal() returned the wrong shape"

    def true_diag(x_stack):
        if offset >= 0:
            x_stack_diag = [x_stack[i, i + offset] for i in range(diag_size)]
        else:
            x_stack_diag = [x_stack[i - offset, i] for i in range(diag_size)]
        return asarray(x_stack_diag, dtype=x.dtype)

    _test_stacks(linalg.diagonal, x, **kw, res=res, dims=1, true_val=true_diag)

@pytest.mark.skip(reason="Inputs need to be restricted")  # TODO
@pytest.mark.xp_extension('linalg')
@given(x=symmetric_matrices(finite=True))
def test_eigh(x):
    res = linalg.eigh(x)

    _test_namedtuple(res, ['eigenvalues', 'eigenvectors'], 'eigh')

    eigenvalues = res.eigenvalues
    eigenvectors = res.eigenvectors

    assert eigenvalues.dtype == x.dtype, "eigh().eigenvalues did not return the correct dtype"
    assert eigenvalues.shape == x.shape[:-1], "eigh().eigenvalues did not return the correct shape"

    assert eigenvectors.dtype == x.dtype, "eigh().eigenvectors did not return the correct dtype"
    assert eigenvectors.shape == x.shape, "eigh().eigenvectors did not return the correct shape"

    _test_stacks(lambda x: linalg.eigh(x).eigenvalues, x,
                 res=eigenvalues, dims=1)
    _test_stacks(lambda x: linalg.eigh(x).eigenvectors, x,
                 res=eigenvectors, dims=2)

    # TODO: Test that res actually corresponds to the eigenvalues and
    # eigenvectors of x

@pytest.mark.xp_extension('linalg')
@given(x=symmetric_matrices(finite=True))
def test_eigvalsh(x):
    res = linalg.eigvalsh(x)

    assert res.dtype == x.dtype, "eigvalsh() did not return the correct dtype"
    assert res.shape == x.shape[:-1], "eigvalsh() did not return the correct shape"

    _test_stacks(linalg.eigvalsh, x, res=res, dims=1)

    # TODO: Should we test that the result is the same as eigh(x).eigenvalues?

    # TODO: Test that res actually corresponds to the eigenvalues of x

@pytest.mark.xp_extension('linalg')
@given(x=invertible_matrices())
def test_inv(x):
    res = linalg.inv(x)

    assert res.shape == x.shape, "inv() did not return the correct shape"
    assert res.dtype == x.dtype, "inv() did not return the correct dtype"

    _test_stacks(linalg.inv, x, res=res)

    # TODO: Test that the result is actually the inverse

@given(
    *two_mutual_arrays(dh.numeric_dtypes)
)
def test_matmul(x1, x2):
    # TODO: Make this also test the @ operator
    if (x1.shape == () or x2.shape == ()
        or len(x1.shape) == len(x2.shape) == 1 and x1.shape != x2.shape
        or len(x1.shape) == 1 and len(x2.shape) >= 2 and x1.shape[0] != x2.shape[-2]
        or len(x2.shape) == 1 and len(x1.shape) >= 2 and x2.shape[0] != x1.shape[-1]
        or len(x1.shape) >= 2 and len(x2.shape) >= 2 and x1.shape[-1] != x2.shape[-2]):
        # The spec doesn't specify what kind of exception is used here. Most
        # libraries will use a custom exception class.
        ph.raises(Exception, lambda: _array_module.matmul(x1, x2),
               "matmul did not raise an exception for invalid shapes")
        return
    else:
        res = _array_module.matmul(x1, x2)

    ph.assert_dtype("matmul", [x1.dtype, x2.dtype], res.dtype)

    if len(x1.shape) == len(x2.shape) == 1:
        assert res.shape == ()
    elif len(x1.shape) == 1:
        assert res.shape == x2.shape[:-2] + x2.shape[-1:]
        _test_stacks(_array_module.matmul, x1, x2, res=res, dims=1)
    elif len(x2.shape) == 1:
        assert res.shape == x1.shape[:-1]
        _test_stacks(_array_module.matmul, x1, x2, res=res, dims=1)
    else:
        stack_shape = sh.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        assert res.shape == stack_shape + (x1.shape[-2], x2.shape[-1])
        _test_stacks(_array_module.matmul, x1, x2, res=res)

matrix_norm_shapes = shared(matrix_shapes())

@pytest.mark.xp_extension('linalg')
@given(
    x=finite_matrices(),
    kw=kwargs(keepdims=booleans(),
              ord=sampled_from([-float('inf'), -2, -2, 1, 2, float('inf'), 'fro', 'nuc']))
)
def test_matrix_norm(x, kw):
    res = linalg.matrix_norm(x, **kw)

    keepdims = kw.get('keepdims', False)
    # TODO: Check that the ord values give the correct norms.
    # ord = kw.get('ord', 'fro')

    if keepdims:
        expected_shape = x.shape[:-2] + (1, 1)
    else:
        expected_shape = x.shape[:-2]
    assert res.shape == expected_shape, f"matrix_norm({keepdims=}) did not return the correct shape"
    assert res.dtype == x.dtype, "matrix_norm() did not return the correct dtype"

    _test_stacks(linalg.matrix_norm, x, **kw, dims=2 if keepdims else 0,
                 res=res)

matrix_power_n = shared(integers(-1000, 1000), key='matrix_power n')
@pytest.mark.xp_extension('linalg')
@given(
    # Generate any square matrix if n >= 0 but only invertible matrices if n < 0
    x=matrix_power_n.flatmap(lambda n: invertible_matrices() if n < 0 else
                             xps.arrays(dtype=xps.floating_dtypes(),
                                        shape=square_matrix_shapes)),
    n=matrix_power_n,
)
def test_matrix_power(x, n):
    res = linalg.matrix_power(x, n)

    assert res.shape == x.shape, "matrix_power() did not return the correct shape"
    assert res.dtype == x.dtype, "matrix_power() did not return the correct dtype"

    if n == 0:
        true_val = lambda x: _array_module.eye(x.shape[0], dtype=x.dtype)
    else:
        true_val = None
    # _test_stacks only works with array arguments
    func = lambda x: linalg.matrix_power(x, n)
    _test_stacks(func, x, res=res, true_val=true_val)

@pytest.mark.xp_extension('linalg')
@given(
    x=finite_matrices(shape=rtol_shared_matrix_shapes),
    kw=kwargs(rtol=rtols)
)
def test_matrix_rank(x, kw):
    linalg.matrix_rank(x, **kw)

@given(
    x=xps.arrays(dtype=dtypes, shape=matrix_shapes()),
)
def test_matrix_transpose(x):
    res = _array_module.matrix_transpose(x)
    true_val = lambda a: _array_module.asarray([[a[i, j] for i in
                                                range(a.shape[0])] for j in
                                                range(a.shape[1])],
                                               dtype=a.dtype)
    shape = list(x.shape)
    shape[-1], shape[-2] = shape[-2], shape[-1]
    shape = tuple(shape)
    assert res.shape == shape, "matrix_transpose() did not return the correct shape"
    assert res.dtype == x.dtype, "matrix_transpose() did not return the correct dtype"

    _test_stacks(_array_module.matrix_transpose, x, res=res, true_val=true_val)

@pytest.mark.xp_extension('linalg')
@given(
    *two_mutual_arrays(dtypes=dh.numeric_dtypes,
                       two_shapes=tuples(one_d_shapes, one_d_shapes))
)
def test_outer(x1, x2):
    # outer does not work on stacks. See
    # https://github.com/data-apis/array-api/issues/242.
    res = linalg.outer(x1, x2)

    shape = (x1.shape[0], x2.shape[0])
    assert res.shape == shape, "outer() did not return the correct shape"
    assert res.dtype == dh.result_type(x1.dtype, x2.dtype), "outer() did not return the correct dtype"

    if 0 in shape:
        true_res = _array_module.empty(shape, dtype=res.dtype)
    else:
        true_res = _array_module.asarray([[x1[i]*x2[j]
                                           for j in range(x2.shape[0])]
                                          for i in range(x1.shape[0])],
                                         dtype=res.dtype)

    assert_exactly_equal(res, true_res)

@pytest.mark.xp_extension('linalg')
@given(
    x=finite_matrices(shape=rtol_shared_matrix_shapes),
    kw=kwargs(rtol=rtols)
)
def test_pinv(x, kw):
    linalg.pinv(x, **kw)

@pytest.mark.xp_extension('linalg')
@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=matrix_shapes()),
    kw=kwargs(mode=sampled_from(['reduced', 'complete']))
)
def test_qr(x, kw):
    res = linalg.qr(x, **kw)
    mode = kw.get('mode', 'reduced')

    M, N = x.shape[-2:]
    K = min(M, N)

    _test_namedtuple(res, ['Q', 'R'], 'qr')
    Q = res.Q
    R = res.R

    assert Q.dtype == x.dtype, "qr().Q did not return the correct dtype"
    if mode == 'complete':
        assert Q.shape == x.shape[:-2] + (M, M), "qr().Q did not return the correct shape"
    else:
        assert Q.shape == x.shape[:-2] + (M, K), "qr().Q did not return the correct shape"

    assert R.dtype == x.dtype, "qr().R did not return the correct dtype"
    if mode == 'complete':
        assert R.shape == x.shape[:-2] + (M, N), "qr().R did not return the correct shape"
    else:
        assert R.shape == x.shape[:-2] + (K, N), "qr().R did not return the correct shape"

    _test_stacks(lambda x: linalg.qr(x, **kw).Q, x, res=Q)
    _test_stacks(lambda x: linalg.qr(x, **kw).R, x, res=R)

    # TODO: Test that Q is orthonormal

    # Check that R is upper-triangular.
    assert_exactly_equal(R, _array_module.triu(R))

@pytest.mark.xp_extension('linalg')
@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=square_matrix_shapes),
)
def test_slogdet(x):
    res = linalg.slogdet(x)

    _test_namedtuple(res, ['sign', 'logabsdet'], 'slotdet')

    sign, logabsdet = res

    assert sign.dtype == x.dtype, "slogdet().sign did not return the correct dtype"
    assert sign.shape == x.shape[:-2], "slogdet().sign did not return the correct shape"
    assert logabsdet.dtype == x.dtype, "slogdet().logabsdet did not return the correct dtype"
    assert logabsdet.shape == x.shape[:-2], "slogdet().logabsdet did not return the correct shape"


    _test_stacks(lambda x: linalg.slogdet(x).sign, x,
                 res=sign, dims=0)
    _test_stacks(lambda x: linalg.slogdet(x).logabsdet, x,
                 res=logabsdet, dims=0)

    # Check that when the determinant is 0, the sign and logabsdet are (0,
    # -inf).
    # TODO: This test does not necessarily hold exactly. Update it to test it
    # approximately.
    # d = linalg.det(x)
    # zero_det = equal(d, zero(d.shape, d.dtype))
    # assert_exactly_equal(sign[zero_det], zero(sign[zero_det].shape, x.dtype))
    # assert_exactly_equal(logabsdet[zero_det], -infinity(logabsdet[zero_det].shape, x.dtype))

    # More generally, det(x) should equal sign*exp(logabsdet), but this does
    # not hold exactly due to floating-point loss of precision.

    # TODO: Test this when we have tests for floating-point values.
    # assert all(abs(linalg.det(x) - sign*exp(logabsdet)) < eps)

def solve_args():
    """
    Strategy for the x1 and x2 arguments to test_solve()

    solve() takes x1, x2, where x1 is any stack of square invertible matrices
    of shape (..., M, M), and x2 is either shape (M,) or (..., M, K),
    where the ... parts of x1 and x2 are broadcast compatible.
    """
    stack_shapes = shared(two_mutually_broadcastable_shapes)
    # Don't worry about dtypes since all floating dtypes are type promotable
    # with each other.
    x1 = shared(invertible_matrices(stack_shapes=stack_shapes.map(lambda pair:
                                                                  pair[0])))

    @composite
    def _x2_shapes(draw):
        end = draw(integers(0, SQRT_MAX_ARRAY_SIZE))
        return draw(stack_shapes)[1] + draw(x1).shape[-1:] + (end,)

    x2_shapes = one_of(x1.map(lambda x: (x.shape[-1],)), _x2_shapes())
    x2 = xps.arrays(dtype=xps.floating_dtypes(), shape=x2_shapes)
    return x1, x2

@pytest.mark.xp_extension('linalg')
@given(*solve_args())
def test_solve(x1, x2):
    linalg.solve(x1, x2)

@pytest.mark.xp_extension('linalg')
@given(
    x=finite_matrices(),
    kw=kwargs(full_matrices=booleans())
)
def test_svd(x, kw):
    res = linalg.svd(x, **kw)
    full_matrices = kw.get('full_matrices', True)

    *stack, M, N = x.shape
    K = min(M, N)

    _test_namedtuple(res, ['U', 'S', 'Vh'], 'svd')

    U, S, Vh = res

    assert U.dtype == x.dtype, "svd().U did not return the correct dtype"
    assert S.dtype == x.dtype, "svd().S did not return the correct dtype"
    assert Vh.dtype == x.dtype, "svd().Vh did not return the correct dtype"

    if full_matrices:
        assert U.shape == (*stack, M, M), "svd().U did not return the correct shape"
        assert Vh.shape == (*stack, N, N), "svd().Vh did not return the correct shape"
    else:
        assert U.shape == (*stack, M, K), "svd(full_matrices=False).U did not return the correct shape"
        assert Vh.shape == (*stack, K, N), "svd(full_matrices=False).Vh did not return the correct shape"
    assert S.shape == (*stack, K), "svd().S did not return the correct shape"

    # The values of s must be sorted from largest to smallest
    if K >= 1:
        assert _array_module.all(S[..., :-1] >= S[..., 1:]), "svd().S values are not sorted from largest to smallest"

    _test_stacks(lambda x: linalg.svd(x, **kw).U, x, res=U)
    _test_stacks(lambda x: linalg.svd(x, **kw).S, x, dims=1, res=S)
    _test_stacks(lambda x: linalg.svd(x, **kw).Vh, x, res=Vh)

@pytest.mark.xp_extension('linalg')
@given(
    x=finite_matrices(),
)
def test_svdvals(x):
    res = linalg.svdvals(x)

    *stack, M, N = x.shape
    K = min(M, N)

    assert res.dtype == x.dtype, "svdvals() did not return the correct dtype"
    assert res.shape == (*stack, K), "svdvals() did not return the correct shape"

    # SVD values must be sorted from largest to smallest
    assert _array_module.all(res[..., :-1] >= res[..., 1:]), "svdvals() values are not sorted from largest to smallest"

    _test_stacks(linalg.svdvals, x, dims=1, res=res)

    # TODO: Check that svdvals() is the same as svd().s.


@given(
    dtypes=mutually_promotable_dtypes(dtypes=dh.numeric_dtypes),
    shape=shapes(),
    data=data(),
)
def test_tensordot(dtypes, shape, data):
    # TODO: vary shapes, vary contracted axes, test different axes arguments
    x1 = data.draw(xps.arrays(dtype=dtypes[0], shape=shape), label="x1")
    x2 = data.draw(xps.arrays(dtype=dtypes[1], shape=shape), label="x2")

    out = xp.tensordot(x1, x2, axes=len(shape))

    ph.assert_dtype("tensordot", dtypes, out.dtype)
    # TODO: assert shape and elements


@pytest.mark.xp_extension('linalg')
@given(
    x=xps.arrays(dtype=xps.numeric_dtypes(), shape=matrix_shapes()),
    # offset may produce an overflow if it is too large. Supporting offsets
    # that are way larger than the array shape isn't very important.
    kw=kwargs(offset=integers(-MAX_ARRAY_SIZE, MAX_ARRAY_SIZE))
)
def test_trace(x, kw):
    res = linalg.trace(x, **kw)

    # TODO: trace() should promote in some cases. See
    # https://github.com/data-apis/array-api/issues/202. See also the dtype
    # argument to sum() below.

    # assert res.dtype == x.dtype, "trace() returned the wrong dtype"

    n, m = x.shape[-2:]
    offset = kw.get('offset', 0)
    assert res.shape == x.shape[:-2], "trace() returned the wrong shape"

    def true_trace(x_stack):
        # Note: the spec does not specify that offset must be within the
        # bounds of the matrix. A large offset should just produce a size 0
        # diagonal in the last dimension (trace 0). See test_diagonal().
        if offset < 0:
            diag_size = min(n, m, max(n + offset, 0))
        elif offset == 0:
            diag_size = min(n, m)
        else:
            diag_size = min(n, m, max(m - offset, 0))

        if offset >= 0:
            x_stack_diag = [x_stack[i, i + offset] for i in range(diag_size)]
        else:
            x_stack_diag = [x_stack[i - offset, i] for i in range(diag_size)]
        return _array_module.sum(asarray(x_stack_diag, dtype=x.dtype), dtype=x.dtype)

    _test_stacks(linalg.trace, x, **kw, res=res, dims=0, true_val=true_trace)


@given(
    dtypes=mutually_promotable_dtypes(dtypes=dh.numeric_dtypes),
    shape=shapes(min_dims=1),
    data=data(),
)
def test_vecdot(dtypes, shape, data):
    # TODO: vary shapes, test different axis arguments
    x1 = data.draw(xps.arrays(dtype=dtypes[0], shape=shape), label="x1")
    x2 = data.draw(xps.arrays(dtype=dtypes[1], shape=shape), label="x2")
    kw = data.draw(kwargs(axis=just(-1)))

    out = xp.vecdot(x1, x2, **kw)

    ph.assert_dtype("vecdot", dtypes, out.dtype)
    # TODO: assert shape and elements


@pytest.mark.xp_extension('linalg')
@given(
    x=xps.arrays(dtype=xps.floating_dtypes(), shape=shapes()),
    kw=kwargs(axis=todo, keepdims=todo, ord=todo)
)
def test_vector_norm(x, kw):
    # res = linalg.vector_norm(x, **kw)
    pass
