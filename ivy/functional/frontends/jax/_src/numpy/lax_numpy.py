# global
import ivy
import ivy.functional.frontends.jax as jax_frontend
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def _rewriting_take(
    arr, idx, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None
):
    return ivy.get_item(arr, idx)


class _IndexUpdateHelper:
    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return _IndexUpdateRef(self.array, index)

    def __setitem__(self, index):
        return _IndexUpdateRef(self.array, index)

    def __repr__(self):
        return f"_IndexUpdateHelper({repr(self.array)})"


class _IndexUpdateRef:
    __slots__ = ("array", "index")

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"_IndexUpdateRef({repr(self.array)}, {repr(self.index)})"

    def get(
        self, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None
    ):
        return _rewriting_take(
            self.array,
            self.index,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
            fill_value=fill_value,
        )

    def set(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        ret = ivy.copy_array(self.array)  # break inplace op
        if hasattr(values, "ivy_array"):
            ret[self.index] = values.ivy_array
        else:
            ret[self.index] = values
        return jax_frontend.Array(ret)
