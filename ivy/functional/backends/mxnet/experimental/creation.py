from typing import Union, Optional, Tuple


def triu_indices(
    n_rows: int, n_cols: Optional[int] = None, k: int = 0, /, *, device: str
) -> Tuple[Union[(None, tf.Variable)]]:
    raise NotImplementedError("mxnet.triu_indices Not Implemented")


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.kaiser_window Not Implemented")


def kaiser_bessel_derived_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.kaiser_bessel_derived_window Not Implemented")


def vorbis_window(
    window_length: Union[(None, tf.Variable)],
    *,
    dtype: None = tf.dtypes.float32,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.vorbis_window Not Implemented")


def hann_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.hann_window Not Implemented")


def tril_indices(
    n_rows: int, n_cols: Optional[int] = None, k: int = 0, /, *, device: str
) -> Tuple[(Union[(None, tf.Variable)], ...)]:
    raise NotImplementedError("mxnet.tril_indices Not Implemented")


def frombuffer(
    buffer: bytes,
    dtype: Optional[None] = float,
    count: Optional[int] = (-1),
    offset: Optional[int] = 0,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.frombuffer Not Implemented")
