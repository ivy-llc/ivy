from .tensorflow__helpers import tensorflow__handle_padding_shape
from .tensorflow__helpers import tensorflow_get_item
from .tensorflow__helpers import tensorflow_pad_1
from .tensorflow__helpers import tensorflow_set_item


def tensorflow_pad(input, pad, mode="constant", value=0):
    if any([(pad_value < 0) for pad_value in pad]):
        pad = list(pad)
        slices = []
        for n in reversed(range(len(pad) // 2)):
            i = n * 2
            j = i + 1
            start = None
            stop = None
            if tensorflow_get_item(pad, i) < 0:
                start = -tensorflow_get_item(pad, i)
                pad = tensorflow_set_item(pad, i, 0)
            if tensorflow_get_item(pad, j) < 0:
                stop = tensorflow_get_item(pad, j)
                pad = tensorflow_set_item(pad, j, 0)
            slices.append(slice(start, stop))
        ndim = len(input.shape)
        while len(slices) < ndim:
            slices.insert(0, slice(None))
        input = tensorflow_get_item(input, tuple(slices))
    value = 0 if value is None else value
    mode_dict = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    if mode not in mode_dict:
        raise ValueError(f"Unsupported padding mode: {mode}")
    pad = tensorflow__handle_padding_shape(pad, len(input.shape), mode)
    return tensorflow_pad_1(
        input, pad, mode=tensorflow_get_item(mode_dict, mode), constant_values=value
    )
