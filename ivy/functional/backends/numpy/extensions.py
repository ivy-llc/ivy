import logging
import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_coo_not_csr,
)


def is_native_sparse_array(x):
    """Numpy does not support sparse arrays natively."""
    return False


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None
):
    ivy.assertions.check_exists(
        data,
        inverse=True,
        message="data cannot be specified, Numpy does not support sparse \
        array natively",
    )
    if _is_coo_not_csr(
        coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
    else:
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
    logging.warning("Numpy does not support sparse array natively, None is returned.")
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    logging.warning(
        "Numpy does not support sparse array natively, None is returned for \
        indices, values and shape."
    )
    return None, None, None


def conv_transpose(
    x: np.ndarray,
    filters: Union[int, float],
    strides: Union[int, float],
    padding: Union[int, float],
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NCDHW",
    dilations: int = 1,
    /,
    *,
    dtype: np.dtype,
    device: str,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return _to_device(
        np.nn.conv_transpose(x, filters, strides, padding, output_shape, data_format, dilations, dtype=dtype
    ), device=device)
