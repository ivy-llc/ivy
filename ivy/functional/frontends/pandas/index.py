import ivy
import ivy.functional.frontends.pandas.series as series


class Index:
    def __init__(self, data, dtype=None, copy=False, name=None, tupleize_cols=True):
        self.index = data
        if not isinstance(data, ivy.Array):
            self.index_array = ivy.array(data, dtype=dtype)
        else:
            self.index_array = data
        self.dtype = dtype
        self.name = name
        self.copy = copy
        self.tupleize_cols = tupleize_cols

    def __repr__(self):
        return f"Index {self.index_array.to_list()}"

    def __len__(self):
        return len(self.index_array)

    @property
    def ndim(self):
        return self.index_array.ndim

    @property
    def size(self):
        return self.index_array.size

    @property
    def array(self):
        return self.index_array

    @property
    def shape(self):
        return tuple(self.index_array.shape)

    @property
    def has_duplicates(self):
        return not self.is_unique()

    def unique(self, level=None):
        # todo handle level with mutliindexer
        self.index_array = ivy.unique_values(self)
        return Index(self.index_array, dtype=self.dtype, copy=self.copy, name=self.name)

    def is_unique(self):
        uniques = ivy.unique_values(self)
        return len(uniques) == len(self.index_array)

    def to_list(self):
        return self.index_array.to_list()

    def to_numpy(self, dtype=None, copy=False, na_value=ivy.nan, **kwargs):
        if dtype:
            return self.index_array.astype(dtype).to_numpy(copy=copy)
        return self.index_array.to_numpy(copy=copy)

    def to_series(self, index=None, name=None):
        if index is None:
            index = self.index_array
        return series.Series(index, index=index, name=name)

    def min(self, axis=None, skipna=True, *args, **kwargs):
        return self.index_array.min()

    def max(self, axis=None, skipna=True, *args, **kwargs):
        return self.index_array.max()

    def isin(self, values, level=None):
        # todo handle level with mutliindexer
        return ivy.isin(self.index_array, values)
