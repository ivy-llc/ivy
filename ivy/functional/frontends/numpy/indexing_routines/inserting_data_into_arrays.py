import ivy

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)
import ivy.functional.frontends.numpy as np_frontend


@to_ivy_arrays_and_back
def fill_diagonal(a, val, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not ivy.all(ivy.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + ivy.sum(ivy.cumprod(a.shape[:-1]))

    # Write the value out into the diagonal.
    shape = a.shape
    temp = ivy.flatten(a)
    temp[:end:step] = val
    a = ivy.reshape(temp, shape)


class AxisConcatenator:
    # allow ma.mr_ to override this
    concatenate = staticmethod(np_frontend.concatenate)
    makemat = staticmethod(np_frontend.matrix)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # copy attributes, since they can be overridden in the first argument
        trans1d = self.trans1d
        ndmin = self.ndmin
        matrix = self.matrix
        axis = self.axis

        objs = []
        # dtypes or scalars for weak scalar handling in result_type
        result_type_objs = []

        for k, item in enumerate(key):
            scalar = False
            if isinstance(item, slice):
                step = item.step
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if ivy.is_complex_dtype(step):
                    size = int(abs(step))
                    newobj = np_frontend.linspace(start, stop, num=size).ivy_array
                else:
                    newobj = np_frontend.arange(start, stop, step).ivy_array
                if ndmin > 1:
                    newobj = np_frontend.array(
                        newobj, copy=False, ndmin=ndmin
                    ).ivy_array
                    if trans1d != -1:
                        newobj = ivy.swapaxes(newobj, -1, trans1d)
            elif isinstance(item, str):
                if k != 0:
                    raise ValueError("special directives must be the first entry.")
                if item in ("r", "c"):
                    matrix = True
                    col = item == "c"
                    continue
                if "," in item:
                    vec = item.split(",")
                    try:
                        axis, ndmin = [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except Exception as e:
                        raise ValueError(
                            "unknown special directive {!r}".format(item)
                        ) from e
                try:
                    axis = int(item)
                    continue
                except (ValueError, TypeError) as e:
                    raise ValueError("unknown special directive") from e
            elif (ivy.isscalar(item)) or (ivy.is_ivy_array(item) and item.ndim == 0):
                scalar = True
                newobj = item
            else:
                item = ivy.array(item)
                item_ndim = item.ndim
                newobj = np_frontend.array(item, copy=False, ndmin=ndmin).ivy_array
                if trans1d != -1 and item_ndim < ndmin:
                    k2 = ndmin - item_ndim
                    k1 = trans1d
                    if k1 < 0:
                        k1 += k2 + 1
                    defaxes = list(range(ndmin))
                    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
                    newobj = np_frontend.transpose(newobj, axes=axes).ivy_array

            objs.append(newobj)
            if scalar:
                result_type_objs.append(item)
            else:
                result_type_objs.append(newobj.dtype)

        # Ensure that scalars won't up-cast unless warranted, for 0, drops
        # through to error in concatenate.
        if len(result_type_objs) != 0:
            if len(result_type_objs) > 1:
                final_dtype = ivy.result_type(*result_type_objs)
            else:
                final_dtype = ivy.result_type(result_type_objs[0], result_type_objs[0])
            # concatenate could do cast, but that can be overridden:
            objs = [
                np_frontend.array(
                    obj, copy=False, ndmin=ndmin, dtype=final_dtype
                ).ivy_array
                for obj in objs
            ]

        res = self.concatenate(tuple(objs), axis=axis)

        if matrix:
            oldndim = res.ndim
            res = self.makemat(res)
            if oldndim == 1 and col:
                res = res.T
        return res

    def __len__(self):
        return 0


class RClass(AxisConcatenator):
    def __init__(self):
        super().__init__(0)


r_ = RClass()


class CClass(AxisConcatenator):
    def __init__(self):
        super().__init__(-1, ndmin=2, trans1d=0)


c_ = CClass()
