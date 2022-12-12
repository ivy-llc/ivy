# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

"""""
Input:
- X array or matrix
- Y array or matrix (optional)
- bias: divide by (N-1) if false, otherwise divide by N. Default is False
- dtype: data type of result
- fweights: 1-D array of frequency weights
- aweights: 1-D array of observation vector weights

Output:
Covariance of the input.

"""


@to_ivy_arrays_and_back
def cov(
    x,
    y=None,
    bias=False,
    dtype=None,
    fweights=None,
    aweights=None,
    ddof=None
):
    # check if inputs are valid
    input_check = ivy.valid_dtype(dtype) and x.ndim in [0, 1]

    if input_check:
        x = ivy.array(x)
        x = x.stack([], axis=0)
        # if two input arrays are given
        if ivy.exists(y) and y.ndim > 0:
            x = x.stack(ivy.array(y), axis=0)

        # compute the weights array
        w = None
        # if weights are 1D and positive
        if ivy.exists(fweights):
            if fweights.ndim < 2 and not fweights.min(keepdims=True)[0] > 0:
                w = ivy.array(fweights)
        if ivy.exists(aweights):
            if aweights.ndim < 2 and not aweights.min(keepdims=True)[0] > 0:
                w = w.multiply(aweights) if ivy.exists(w) else ivy.array(aweights)

            # if w exists, use weighted average
            xw = x.multiply(w)
            w_sum = ivy.sum(w)
            average = ivy.stable_divide(ivy.sum(xw, axis=1) , w_sum)
        else:
            # else compute arithmetic average
            average = ivy.mean(x, axis=1)

        # compute the normalization
        if ddof is None:
            ddof = 1 if bias == 0 else 0

        if w is None:
            norm = x.shape[0] - ddof
        elif ddof == 0:
            norm = w_sum
        elif aweights is None:
            norm = w_sum - ddof
        else:
            norm = w_sum - ivy.stable_divide(ddof * ivy.sum(w * aweights), w_sum)

        # compute residuals from average
        x -= average[:]
        # compute transpose matrix
        x_t = ivy.matrix_transpose(x * w) if ivy.exists(w) else ivy.matrix_transpose(x)
        # compute covariance matrix
        c = ivy.stable_divide(ivy.matmul(x, x_t), norm).astype(dtype)

        return c
