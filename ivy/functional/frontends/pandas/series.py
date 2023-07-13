import ivy
import numpy as np


class Series:
    def __init__(
        self, data, index=None, dtype=None, name=None, copy=False, fastpath=False
    ):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
            self.array = data

        data_is_array = isinstance(data, (ivy.Array, np.ndarray))

        if data_is_array:
            assert data.ndim == 1, "Series Data must be 1-dimensional"

        # setup a default index if none provided
        if index is None:
            if data_is_array or isinstance(data, (list, tuple)):
                index = ivy.arange(len(data)).tolist()
            elif isinstance(data, dict):
                index = list(data.keys())

        if data_is_array:
            if len(data) != len(index):
                raise ValueError(
                    f"Length of values {len(data)} does not match length of index"
                    f" {len(index)}"
                )
            self.data = data
            self.index = index
            self.array = data

        elif isinstance(data, dict):
            pass
        elif isinstance(data, (list, tuple)):
            pass
        elif isinstance(data, (int, float, str)):
            pass

        self.index = index
        self.dtype = dtype
        self.name = name

    def __repr__(self):
        series_name = f"{self.name} " if self.name is not None else ""
        return f"frontends.pandas.Series {series_name}({self.array.to_list()}, index={self.index})"
