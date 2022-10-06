# global
import torch

# local
import ivy


def can_cast(from_, to):
    res = 0
    native_dtype_dict = {
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bool': torch.bool,
    }
    if isinstance(from_, str) is False and isinstance(to, str) is False:
        key_list = list(native_dtype_dict.keys())
        val_list = list(native_dtype_dict.values())
        d1 = key_list[val_list.index(from_)]
        d2 = key_list[val_list.index(to)]
        res = ivy.can_cast(d1, d2)

    else:
        res = ivy.can_cast(from_, to)
    return res
