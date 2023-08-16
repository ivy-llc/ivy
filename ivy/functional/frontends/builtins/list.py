import ivy
from ivy.functional.frontends.builtins.func_wrapper import _to_ivy_array
import ivy.functional.frontends.builtins as builtins_frontend


class list(list):
    def __init__(self, iterable=None):
        super().__init__()
        self._ivy_array = (
            ivy.array(iterable if iterable is not None else [])
            if not isinstance(iterable, ivy.Array)
            else iterable
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    # Dunderscore methods #
    # ------------------- #

    def __repr__(self):
        return str(self.ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.builtins.List"
        )

    def __len__(self):
        return len(self.ivy_array)

    def __getitem__(self, item):
        item = _to_ivy_array(item)
        return ivy.get_item(self.ivy_array, item)

    def __iter__(self):
        iter_len = self.__len__()
        if iter_len == 0:
            raise TypeError("Iteration over 0-d list is not supported!")

        for i in builtins_frontend.range(iter_len):
            yield self[i]

    def __str__(self):
        return f"{self.ivy_array}"

    def __abs__(self):
        return builtins_frontend.abs(self)

    # Methods #
    # ------- #

    def pop(self, key=-1):
        key = _to_ivy_array(key)

        if key < 0:
            key += self.__len__()

        temp = self[key]
        self.ivy_array = ivy.concat([self[:key], self[key + 1 :]])
        return temp
