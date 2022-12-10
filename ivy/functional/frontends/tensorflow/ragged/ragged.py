import ivy


# TODO: Align behavior with tensorflow, modify so that the elements of the raggedTensor
#  object are of type EagerTensor
# ensure that the values and row_splits are of type EagerTensor too
# add more initializer methods


class RaggedTensor:
    def __init__(self, values, row_partition, internal=False, data=None):
        if not internal:
            raise ivy.exceptions.IvyException(
                "RaggedTensor constructor is private; please use one of the "
                "factory methods instead "
                "(e.g., RaggedTensor.from_row_lengths())"
            )
        self._values = values
        self.data = data
        self._row_partition = row_partition

    @classmethod
    def from_row_splits(cls, values, row_splits, name=None, validate=True):
        # TODO : modify this, if necessary, to accept raggedTensor inputs too

        if values.shape[0] != row_splits[-1] or row_splits[0] != 0:
            if values.shape[0] != row_splits[-1]:
                ivy.exceptions.IvyException(
                    "first dimension of shape of values should be equal to the"
                    " last dimension of row_splits"
                )
            else:
                ivy.exceptions.IvyException(
                    "first value of row_splits should be equal to zero."
                )
        data = [
            values[row_splits[i] : row_splits[i + 1], :]
            for i in range(len(row_splits) - 1)
        ]

        return cls(values=values, row_partition=row_splits, internal=True, data=data)

    @classmethod
    def from_row_lengths(
        cls,
        values,
        row_lengths,
        name=None,
    ):
        # TODO : modify this, if necessary, to accept raggedTensor inputs too
        if sum(row_lengths) != values.shape[0]:
            ivy.exceptions.IvyException(
                "first dimension of values should be equal to sum(row_lengths) "
            )
        data = []
        z = 0
        for length in row_lengths:
            temp = []
            for i in range(length):
                temp.append(values[z, :])
                z += 1

            data.append(ivy.asarray(temp))

        # data =[[values[0+i,:] for i in range(length)]
        #   for length in row_lengths]
        return cls(values=values, row_partition=row_lengths, internal=True, data=data)

    @classmethod
    def from_value_rowids(
        cls,
        values,
        value_rowids,
        nrows=None,
        name=None,
    ):
        if not nrows:
            nrows = value_rowids[-1] + 1
        data = []
        for row in range(nrows):
            temp = []
            for i in range(len(values)):
                if value_rowids[i] == row:
                    temp.append(values[i, :])
            data.append(ivy.asarray(temp))

        # data= [[values[i,:] for i in range(len(values)) if value_rowids[i] == row]
        #   for row in range(nrows)]
        return cls(values=values, row_partition=value_rowids, internal=True, data=data)

    @classmethod
    def from_row_starts(
        cls,
        values,
        row_starts,
        name=None,
    ):
        # TODO since row_starts will be a tensor try appending using concat after
        #  ensuring row_starts
        # is a tensor
        row_starts.append(len(values))
        return cls.from_row_splits(values, row_starts)

    def to_list(self):
        vals = []
        for i in self:
            if isinstance(i, RaggedTensor):
                vals.append(i.to_list())
            else:
                vals.append(ivy.to_list(i))
        return vals

    @property
    def values(self):
        return self._values

    @property
    def flat_values(self):
        values = self.values
        while isinstance(values, RaggedTensor):
            values = values.values
        return values

    @property
    def row_splits(self):
        return self._row_partition

    @property
    def nested_row_splits(self):
        rt_nested_splits = [self.row_splits]
        rt_values = self.values
        while isinstance(rt_values, RaggedTensor):
            rt_nested_splits.append(rt_values.row_splits)
            rt_values = rt_values.values
        return tuple(rt_nested_splits)
