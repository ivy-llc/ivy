import ivy
import numpy as np

class Series:
    def __init__(self,
                 data,
                 index=None,
                 dtype=None,
                 name=None,
                 copy=False,
                 fastpath=False):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        data_is_array = isinstance(data, (ivy.Array, np.ndarray))

        if data_is_array:
            assert data.ndim == 1

        if index is None:
            if data_is_array or isinstance(data, (list, tuple)):
                index = ivy.arange(len(data))
            elif isinstance(data, dict):
                index = list(data.keys())

        self.data = data
        self.index = index
        self.dtype = dtype
        self.name = name
