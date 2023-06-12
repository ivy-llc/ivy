# local
from .base import NestedArrayBase


class NestedArray(NestedArrayBase):
    def __init__(self, data, dtype, device, internal=False):
        NestedArrayBase.__init__(self, data, dtype, device, internal)

    @classmethod
    def from_row_lengths(cls, values, row_lengths):
        ivy_arrays = list()
        for i in range(len(row_lengths)):
            ivy_arrays.append(values[: row_lengths[i]])
            values = values[row_lengths[i] :]
        return cls.nested_array(ivy_arrays)

    @classmethod
    def from_row_splits(cls, values, row_split):
        row_lengths = list()
        for i in range(1, len(row_split)):
            row_lengths.append(row_split[i] - row_split[i - 1])
        return cls.from_row_lengths(values, row_lengths)
