# local
from .base import NestedArrayBase


class NestedArray(NestedArrayBase):
    def __init__(self, data, nested_rank, inner_shape, dtype, device, internal=False):
        NestedArrayBase.__init__(
            self, data, nested_rank, inner_shape, dtype, device, internal
        )

    @classmethod
    def from_row_lengths(cls, values, row_lengths):
        ivy_arrays = []
        for i in range(len(row_lengths)):
            ivy_arrays.append(values[: row_lengths[i]])
            values = values[row_lengths[i] :]
        return cls.nested_array(ivy_arrays)

    @classmethod
    def from_row_splits(cls, values, row_splits):
        row_lengths = []
        for i in range(1, len(row_splits)):
            row_lengths.append(row_splits[i] - row_splits[i - 1])
        return cls.from_row_lengths(values, row_lengths)
