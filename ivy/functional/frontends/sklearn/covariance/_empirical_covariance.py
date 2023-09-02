import ivy
from ivy.functional.frontends.numpy import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def empirical_covariance(X, *, assume_centered = False):
    if X.ndim == 1:
        X = ivy.reshape(X, (1, -1))
    
    if assume_centered:
        covariance = ivy.dot(X.T, X) / X.shape[0]
    else:
        covariance = ivy.cov(X.T, bias = 1)

    if covariance.ndim == 0:
        covariance = ivy.array([[covariance]])

    if ivy.is_complex_dtype(X):
        return covariance.astype(ivy.complex128)

    return covariance.astype(ivy.float64)
