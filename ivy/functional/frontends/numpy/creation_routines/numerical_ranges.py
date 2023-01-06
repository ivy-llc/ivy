# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    outputs_to_numpy_arrays,
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@outputs_to_numpy_arrays
def arange(start, stop=None, step=1, dtype=None, *, like=None):
    return ivy.arange(start, stop, step, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    ret = ivy.linspace(start, stop, num, axis=axis, endpoint=endpoint, dtype=dtype)
    if retstep:
        if endpoint:
            num -= 1
        step = ivy.divide(ivy.subtract(stop, start), num)
        return ret, step
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if not endpoint:
        interval = (stop - start) / num
        stop -= interval
    return ivy.logspace(start, stop, num, base=base, axis=axis, dtype=dtype)


@to_ivy_arrays_and_back
def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    # Todo: add sparse check
    ret = ivy.meshgrid(*xi, indexing=indexing)
    if copy:
        return [ivy.copy_array(x) for x in ret]
    return ret


class nd_grid:
    def __init__(self, sparse=False):
        self.sparse = sparse
        self.grids = []
        self.shapes = []

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = self._split_slice(key)
            ret = ivy.arange(start, stop, step)
            return (
                ivy.native_array(ret, dtype="int64")
                if ivy.is_int_dtype(ret)
                else ivy.native_array(ret, dtype="float64")
            )
        # more than one slice , key is tuple
        self.grids = []
        self.shapes = []
        for k in key:
            start, stop, step = self._split_slice(k)
            ret = ivy.arange(start, stop, step)
            self.grids.append(ret)
            self.shapes.append(ivy.shape(ret)[0])
        self._process_arrays()
        return self._ret_grids()

    def _split_slice(self, slice):
        start = slice.start
        stop = slice.stop
        step = slice.step
        if start is None:
            start = 0
        elif stop is None:
            stop = start
            start = 0
        if isinstance(step, complex):
            step = abs(stop - start) / (int(abs(step)) - 1)
            stop += step
        elif step is None:
            step = 1
        return start, stop, step

    def _process_arrays(self):
        total_arr = len(self.grids)
        current_arr = total_arr
        while current_arr != 0:
            arr = self._shape_array(self.grids[current_arr - 1], current_arr, total_arr)
            if self.sparse:
                self.grids[current_arr - 1] = arr
            else:
                self.grids[current_arr - 1] = arr[0]
            current_arr -= 1

    def _init_array(self, array, current, total):
        rep = 1
        for i in range(current, total):
            rep *= self.shapes[i]
        return ivy.repeat(array, rep, axis=0)

    def _shape_array(self, array, current, total):
        # ogrid
        if self.sparse:
            new_shape = [1] * total
            new_shape[current - 1] = self.shapes[current - 1]
            return ivy.reshape(array, new_shape)
        # mgrid
        if current != total:
            array = self._init_array(array, current, total)
        while current != 1:
            new_shape = [1] + self.shapes[current - 1 : total]
            array = ivy.reshape(array, new_shape)
            array = ivy.repeat(array, self.shapes[current - 2], axis=0)
            current -= 1
        array = ivy.reshape(array, [1] + self.shapes)
        return array

    def _ret_grids(self):
        is_float = False
        for grid in self.grids:
            if ivy.is_float_dtype(grid):
                is_float = True
                break
        # ogrid
        if self.sparse:
            for i in range(0, len(self.grids)):
                self.grids[i] = (
                    ivy.native_array(self.grids[i], dtype="float64")
                    if is_float
                    else ivy.native_array(self.grids[i], dtype="int64")
                )
            return self.grids
        # mgrid
        return (
            ivy.native_array(self.grids, dtype="float64")
            if is_float
            else ivy.native_array(self.grids, dtype="int64")
        )


class MGrid(nd_grid):
    def __init__(self):
        super().__init__(sparse=False)


mgrid = MGrid()


class OGrid(nd_grid):
    def __init__(self):
        super().__init__(sparse=True)


ogrid = OGrid()
