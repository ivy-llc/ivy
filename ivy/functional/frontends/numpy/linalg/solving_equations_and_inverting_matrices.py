# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs


# solve
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def solve(a, b):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.solve(a, b)


# inv
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def inv(a):
    return ivy.inv(a)


# pinv
# TODO: add hermitian functionality
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol=rtol)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.23.0 and below": ("float16", "blfloat16")}, "numpy")
def tensorinv(a, ind=2):
    old_shape = ivy.shape(a)
    prod = 1
    if ind > 0:
        invshape = old_shape[ind:] + old_shape[:ind]
        for k in old_shape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = ivy.reshape(a, shape=(prod, -1))
    ia = ivy.inv(a)
    new_shape = tuple([*invshape])
    return ivy.reshape(ia, shape=new_shape)

@to_ivy_arrays_and_back
def lstsq(a, b, rcond='warn'):
    solution = ivy.matmul(ivy.pinv(a), b)

    svd = ivy.svd(a, compute_uv=False)
    if a.ndim < 2:
        rank = int(not all(a == 0))
    else:
        S = svd[0]
        tol = S.max() * max(a.shape) * ivy.finfo(S.dtype).eps
        rank = ivy.count_nonzero(S > tol, axis=-1).astype(ivy.int64)
    residuals = ivy.sum((b - ivy.matmul(a, solution))**2)
    print("*" * 10)
    print([solution, residuals, rank, svd[0]])
    print("-" * 10)
    return [solution, residuals, rank, svd[0]]




