# local
from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays


class ufunc:
    def __init__(self):
        pass

    @inputs_to_ivy_arrays
    def at(self, indices, b=None, /):
        array = self._ivy_array
        if b:
            for index in indices:
                array[index] = self(array[index], b)
        else:
            for index in indices:
                array[index] = self(array[index])
