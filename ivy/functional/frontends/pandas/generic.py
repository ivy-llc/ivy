import ivy
import numpy as np
import copy as py_copy
from ivy.functional.frontends.pandas.pandas_func_wrappers import outputs_to_self_class
import ivy.functional.frontends.pandas.series as series


class NDFrame:
    def __init__(self, data, index, columns, dtype, name, copy, *args, **kwargs):
        self.name = name
        self.columns = columns
        self.dtype = dtype
        self.copy = copy
        self.orig_data = py_copy.deepcopy(data)

        if ivy.is_native_array(data):
            self.array = ivy.array(data)

        # repeatedly used checks
        data_is_array = isinstance(data, (ivy.Array, np.ndarray))
        data_is_array_or_like = data_is_array or isinstance(data, (list, tuple))

        # setup a default index if none provided
        orig_data_len = len(self.orig_data)
        if index is None:
            if data_is_array_or_like:
                index = ivy.arange(orig_data_len).tolist()
            elif isinstance(data, dict):
                index = list(data.keys())
            elif isinstance(data, series.Series):
                index = data.index
        elif isinstance(data, dict) and len(index) > orig_data_len:
            for i in index:
                if i not in data:
                    data[i] = ivy.nan

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
        elif isinstance(data, series.Series):
            self.array = data.array
            self.index = index
        elif isinstance(data, str):
            pass  # TODO: implement string series
        else:
            raise TypeError(
                "Data must be one of array, dict, iterables, scalar value or Series. Got"
                f" {type(data)}"
            )

    @property
    def data(self):
        # return underlying data in the original format
        ret = self.array.to_list()
        if isinstance(self.orig_data, tuple):
            ret = tuple(ret)
        elif isinstance(self.orig_data, dict):
            ret = dict(zip(self.orig_data.keys(), ret))
        return ret

    @outputs_to_self_class
    def abs(self):
        return ivy.abs(self.array)

    def to_numpy(self, dtype=None, copy=False, na_value=None):
        ret = self.array.to_numpy()
        if na_value is not None:
            ret = np.where(ret == np.nan, na_value, ret)
        if dtype is not None:
            ret = ret.astype(dtype)
        if copy:
            return ret.copy()
        return ret

    def __array__(self):
        return self.array.to_numpy()

    @outputs_to_self_class
    def __array_wrap__(self, array):
        return array
