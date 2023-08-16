from .generic import NDFrame
import ivy
from .series import Series


class DataFrame(NDFrame):
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        name=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            data,
            index=index,
            dtype=dtype,
            copy=copy,
            name=None,
            columns=None,
            *args,
            **kwargs,
        )
        if isinstance(self.orig_data, dict):
            # if data is a dict the underlying array needs to be extended to match the
            # index as a 2d array
            self.columns = list(self.orig_data.keys()) if columns is None else columns
            array_data = list(self.orig_data.values())
            self.array = ivy.array([array_data for _ in range(len(self.index))])
        elif isinstance(self.orig_data, Series):
            self.array = self.array.expand_dims()
            self.columns = [0]
        elif columns is None:
            self.columns = ivy.arange(self.array.shape[1]).tolist()
        else:
            self.columns = columns

        assert self.array.ndim == 2, "DataFrame Data must be 2-dimensional"

    def __getitem__(self, col):
        # turn labels (strings) into numbered indexing so that self.array columns can
        # be accessed.
        if isinstance(col, (tuple, list)):
            numbered_col = [self.columns.index(i) for i in col]
            return DataFrame(
                self.array[:, numbered_col],
                index=self.index,
                dtype=self.dtype,
                columns=col,
            )
        col = self.columns.index(col)
        return Series(
            self.array[:, col],
            index=self.index,
            dtype=self.dtype,
        )

    def __getattr__(self, item):
        if item in self.columns:
            item_index = self.columns.index(item)
            return Series(
                self.array[:, item_index],
                index=self.index,
                dtype=self.dtype,
            )
        else:
            return super().__getattr__(item)

    def __repr__(self):
        return (
            f"frontends.pandas.DataFrame ({self.array.to_list()}, "
            f"index={self.index}), columns={self.columns})"
        )

    def sum(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0):
        _array = self.array
        if axis is None or axis == "index":
            axis = 0  # due to https://github.com/pandas-dev/pandas/issues/54547. TODO: remove this when fixed
        elif axis == "columns":
            axis = 1
        if min_count > 0:
            if ivy.has_nans(_array):
                number_values = _array.size - ivy.sum(ivy.isnan(_array))
            else:
                number_values = _array.size
            if min_count > number_values:
                return ivy.nan
        if skipna:
            ret = ivy.nansum(_array, axis=axis)
        else:
            ret = _array.sum(axis=axis)
        return Series(ret, index=self.columns if axis in (0, "index") else self.index)
