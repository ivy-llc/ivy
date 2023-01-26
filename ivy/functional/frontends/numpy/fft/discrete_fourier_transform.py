import ivy
import ivy.functional.frontends.numpy as np_frontend

# no implementation of the underlying C functions in Ivy
import numpy.fft._pocketfft_internal as pfi 
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def _raw_fft(a, n, axis, is_real, is_forward, inv_norm):
    axis = ivy.normalize_axis_tuple(axis, a.ndim)
    # the correct function should be normalize_axis_index but is not 
    # currently implemented in the Ivy API
    # normalize_axis_index does the same thing but returns the result as a tuple.
    # In this case, only a tuple with one element should be returned.
    axis = axis[0]

    if n is None:
        n = a.shape[axis]

    fct = 1/inv_norm

    if a.shape[axis] != n:
        s = list(a.shape)
        index = [slice(None)]*len(s)
        if s[axis] > n:
            index[axis] = slice(0, n)
            a = a[tuple(index)]
        else:
            index[axis] = slice(0, s[axis])
            s[axis] = n
            z = ivy.zeros(s, a.dtype.char)
            z[tuple(index)] = a
            a = z

    if axis == a.ndim - 1:
        # use an ndarray to as argument to numpy's fft execute function and 
        # convert result back to an ivy array
        r = ivy.array(pfi.execute(ivy.to_numpy(a), is_real, is_forward, fct))
    else:
        a = np_frontend.swapaxes(a, axis, -1)
        r = ivy.array(pfi.execute(ivy.to_numpy(a), is_real, is_forward, fct))
        r = np_frontend.swapaxes(r, axis, -1)
   
    return r

def _get_backward_norm(n, norm):
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    if norm is None or norm == "backward":
        return n
    elif norm == "ortho":
        return ivy.sqrt(n)
    elif norm == "forward":
        return 1
    raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                     '"ortho" or "forward".')


def ifft(a, n=None, axis=-1, norm=None):
    a = ivy.asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, False, False, inv_norm)
    return output

