from typing import Tuple, Union, Optional


def unique_all(
    x: Union[(None, tf.Variable)], /, *, axis: Optional[int] = None
) -> Tuple[
    (
        Union[(None, tf.Variable)],
        Union[(None, tf.Variable)],
        Union[(None, tf.Variable)],
        Union[(None, tf.Variable)],
    )
]:
    raise NotImplementedError("mxnet.unique_all Not Implemented")


def unique_counts(
    x: Union[(None, tf.Variable)], /
) -> Tuple[(Union[(None, tf.Variable)], Union[(None, tf.Variable)])]:
    raise NotImplementedError("mxnet.unique_counts Not Implemented")


def unique_inverse(
    x: Union[(None, tf.Variable)], /
) -> Tuple[(Union[(None, tf.Variable)], Union[(None, tf.Variable)])]:
    raise NotImplementedError("mxnet.unique_inverse Not Implemented")


def unique_values(
    x: Union[(None, tf.Variable)],
    /,
    *,
    out: Optional[Union[(None, tf.Variable)]] = None,
) -> Union[(None, tf.Variable)]:
    raise NotImplementedError("mxnet.unique_values Not Implemented")
