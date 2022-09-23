import ivy
from typing import Optional
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_data_not_indices_values_and_shape,
    _is_coo_not_csr,
)
import tensorflow as tf
import logging


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


def native_sparse_array(
        data=None,
        *,
        coo_indices=None,
        csr_crow_indices=None,
        csr_col_indices=None,
        values=None,
        dense_shape=None
):
    if _is_data_not_indices_values_and_shape(
            data, coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        ivy.assertions.check_true(
            ivy.is_native_sparse_array(data), message="not a sparse array"
        )
        return data
    elif _is_coo_not_csr(
            coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
        all_coordinates = []
        for i in range(values.shape[0]):
            coordinate = ivy.gather(coo_indices, ivy.array([[i]]))
            coordinate = ivy.reshape(coordinate, (coo_indices.shape[0],))
            all_coordinates.append(coordinate.to_list())
        return tf.SparseTensor(
            indices=all_coordinates, values=values, dense_shape=dense_shape
        )
    else:
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
        logging.warning(
            "Tensorflow does not support CSR sparse array natively. None is returned."
        )
        return None


def native_sparse_array_to_indices_values_and_shape(x):
    if isinstance(x, tf.SparseTensor):
        return x.indices, x.values, x.dense_shape
    raise ivy.exceptions.IvyException("not a SparseTensor")


def _get_backward_norm(n, norm):
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    if norm is None or norm == "backward":
        return 1 / n
    elif norm == "ortho":
        return 1 / ivy.sqrt(n)
    elif norm == "forward":
        return 1
    raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                     '"ortho" or "forward".')


def _handle_axis_ifft(axis, input, n, inv_norm):
    if n is None:
        n = input.shape[axis]

    if input.shape[axis] != n:
        s = list(input.shape)
        index = [slice(None)] * len(s)
        if s[axis] > n:
            index[axis] = slice(0, n)
            input = input[tuple(index)]
        else:
            index[axis] = slice(0, s[axis])
            s[axis] = n
            z = ivy.zeros(s, input.dtype.char)
            z[tuple(index)] = input
            input = z
    return input / inv_norm


def ifft(input: Union[tf.Tensor, tf.Variable], n: Optional[int] = None,
         dim: Optional[int] = None, axis: Optional[int] = None,
         norm: Optional[str] = None, name: Optional[str] = None):
    if not isinstance(ivy.backend(), ivy.get_backend("tensorflow")):
        inv_norm = _get_backward_norm(n, norm)
        input = _handle_axis_ifft(axis, input, n, inv_norm)
    return tf.signal.ifft(input, name=name)
