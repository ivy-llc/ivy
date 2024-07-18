from .tensorflow__helpers import tensorflow_dim
from .tensorflow__helpers import tensorflow_size_2


def tensorflow__calculate_fan_in_and_fan_out(tensor):
    dimensions = tensorflow_dim(tensor)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = tensorflow_size_2(tensor, 1)
    num_output_fmaps = tensorflow_size_2(tensor, 0)
    receptive_field_size = 1
    if tensorflow_dim(tensor) > 2:
        for s in tensor.shape[2:]:
            receptive_field_size = receptive_field_size * s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out
