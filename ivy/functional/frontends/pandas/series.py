import ivy
import numpy as np


class Series:
    def __init__(
        self, data, index=None, dtype=None, name=None, copy=False, fastpath=False
    ):
        self.name = name
        self.orig_data = data
        if ivy.is_native_array(data):
            self.array = ivy.array(data)

        # repeatedly used checks
        data_is_array = isinstance(data, (ivy.Array, np.ndarray))
        data_is_array_or_like = data_is_array or isinstance(data, (list, tuple))

        if data_is_array:
            assert data.ndim == 1, "Series Data must be 1-dimensional"

        # setup a default index if none provided
        if index is None:
            if data_is_array_or_like:
                index = ivy.arange(len(data)).tolist()
                if len(data) != len(index):
                    raise ValueError(
                        f"Length of values {len(data)} does not match length of index"
                        f" {len(index)}"
                    )
            elif isinstance(data, dict):
                index = list(data.keys())

        if data_is_array_or_like:
            self.index = index
            self.array = ivy.array(data)

        elif isinstance(data, dict):
            self.index = index
            self.array = ivy.array(list(data.values()))

        elif isinstance(data, (int, float)):
            if len(index) > 1:
                data = [data] * len(index)
            self.index = index
            self.array = ivy.array(data)
        elif isinstance(data, str):
            pass # TODO: implement string series
        else:
            raise TypeError(f"Data must be one of array, dict, iterables or scalar value, got {type(data)}")

    def __repr__(self):
        series_name = f"{self.name} " if self.name is not None else ""
        return f"frontends.pandas.Series {series_name}({self.array.to_list()}, index={self.index})"

    @property
    def data(self):
        # return underlying data in the original format
        ret = self.array.to_list()
        if isinstance(self.orig_data, tuple):
            ret = tuple(ret)
        elif isinstance(self.orig_data, dict):
            ret = dict(zip(self.orig_data.keys(), ret))
        return ret
