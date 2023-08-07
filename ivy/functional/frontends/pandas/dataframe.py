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
        if isinstance(col, (tuple, list)):
            indexed_col = [self.columns.index(i) for i in col]
            return DataFrame(
                self.array[:, indexed_col],
                index=self.index,
                dtype=self.dtype
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
