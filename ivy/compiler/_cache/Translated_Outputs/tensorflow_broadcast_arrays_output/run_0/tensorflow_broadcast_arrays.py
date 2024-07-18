import tensorflow

from typing import Union


def tensorflow_broadcast_arrays(*arrays: Union[tensorflow.Tensor, tensorflow.Variable]):
    if len(arrays) > 1:
        try:
            desired_shape = tensorflow.broadcast_dynamic_shape(
                tensorflow.shape(arrays[0]), tensorflow.shape(arrays[1])
            )
        except tensorflow.errors.InvalidArgumentError as e:
            raise Exception(e) from e
        if len(arrays) > 2:
            for i in range(2, len(arrays)):
                try:
                    desired_shape = tensorflow.broadcast_dynamic_shape(
                        desired_shape, tensorflow.shape(arrays[i])
                    )
                except tensorflow.errors.InvalidArgumentError as e:
                    raise Exception(e) from e
    else:
        return [arrays[0]]
    result = []
    for tensor in arrays:
        result.append(tensorflow.broadcast_to(tensor, desired_shape))
    return result
