import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from typing import Any

class RaggedTensor:
    def __init__(self, values, row_partition, internal=False, data=None):
        if not internal:
            raise ivy.exceptions.IvyException("RaggedTensor constructor is private; please use one of the factory methods instead (e.g., RaggedTensor.from_row_lengths())")
        self._values=values
        self.data=data
        self._row_partition=row_partition

    @classmethod
    def from_row_splits(cls, values, row_splits, name=None, validate=True):
        values = ivy.reshape(values, -1)
        data=[values[row_splits[i]:row_splits[i + 1]]
          for i in range(len(row_splits) - 1)]

        return cls(values=values,row_partition=row_splits, internal=True, data =data)

    @classmethod
    def from_row_lengths(cls,
        values, row_lengths, name=None,
    ):
        values = ivy.reshape(values, -1)
        data =[[values.pop(0) for i in range(length)]
          for length in row_lengths]
        return cls(values=values, row_partition=row_lengths, internal=True, data=data)

    @classmethod
    def from_value_rowids(cls,
        values, value_rowids, nrows=None, name=None,
    ):
        values=ivy.reshape(values,-1)
        data= [[values[i] for i in range(len(values)) if value_rowids[i] == row]
          for row in range(nrows)]
        return cls(values=values, row_partition=value_rowids, internal=True, data=data)

    @classmethod
    def from_row_starts(cls,
        values, row_starts, name=None,
    ):
        values = ivy.reshape(values, -1)

        return cls(values=values, row_partition=row_starts, internal=True, data=data)





