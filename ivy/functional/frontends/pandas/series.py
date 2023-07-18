from .generic import NDFrame


class Series(NDFrame):
    def __init__(
        self,
        data,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        fastpath=False,
        columns=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            data,
            index,
            columns=None,
            dtype=dtype,
            name=name,
            copy=copy,
            *args,
            **kwargs,
        )
        assert self.array.ndim == 1, "Series Data must be 1-dimensional"

    def __repr__(self):
        series_name = f"{self.name} " if self.name is not None else ""
        return (
            f"frontends.pandas.Series {series_name}({self.array.to_list()},"
            f" index={self.index})"
        )
