# global
from typing import Union, Optional, Tuple

# local
import tensorflow as tf
import numpy as np
import ivy


def _fft_norm(
    x:Union[tf.Tensor, tf.Variable],
    dim: int,
    /,
    *,
    norm: str="backward",
):
    n = tf.constant(x.shape[dim])
    if norm == "backward":
        return x
    elif norm == "ortho":
        return x/tf.sqrt(n)
    elif norm == "forward":
        return x/n
    else :
        raise ivy.exceptions.IvyError(f"Unrecognized normalization mode {norm}")


def fft(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    /,
    *,
    norm: str="backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    if not isinstance(dim,int):
        raise ivy.exceptions.IvyError(f"Expecting <class 'int'> instead of {type(dim)}")
    if n < -len(x.shape) :
        raise ivy.exceptions.IvyError(f"Invalid dim {dim}, expecting ranging from {-len(x.shape)} to {len(x.shape)-1}  ")
    if n is None:
        n = x.shape[dim]
    if not isinstance(n,int):
        raise ivy.exceptions.IvyError(f"Expecting <class 'int'> instead of {type(n)}")
    if n <= 1 :
        raise ivy.exceptions.IvyError(f"Invalid data points {n}, expecting more than 1")
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    if x.shape[dim] != n:
        s = list(x.shape)
        if s[dim] > n:
            index = [slice(None)]*len(s)
            index[dim] = slice(0, n)
            x = x[tuple(index)]
            del index
        else:
            s[dim] = n-s[dim]
            z = tf.zeros(s, x.dtype)
            x = tf.concat([x,z],axis=dim)
        del s
    operation_name = f"{n} points FFT at dim {dim} with {norm} normalization"
    if dim != -1 or dim != len(x.shape)-1:
        permute = [i for i in range(len(x.shape))]
        permute[dim],permute[-1] = permute[-1],permute[dim]
        x = tf.transpose(x,permute)
        ret = tf.signal.fft(x,operation_name)
        x = tf.transpose(x,permute)
        del permute
    else:
        ret = tf.signal.fft(x,operation_name)
    ret = _fft_norm(ret,dim,norm)
    return ret