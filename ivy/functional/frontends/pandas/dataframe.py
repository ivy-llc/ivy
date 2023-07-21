from .generic import NDFrame
import ivy


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

        assert self.array.ndim == 2, "DataFrame Data must be 2-dimensional"

        if columns is None:
            self.columns = ivy.arange(self.array.shape[1]).tolist()
        else:
            self.columns = columns

    def __repr__(self):
        return (
            f"frontends.pandas.DataFrame ({self.array.to_list()}, "
            f"index={self.index}), columns={self.columns})"
        )
