# global

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class ndarray:
    def __init__(self, object):
        if ivy.is_native_array(object):
            object = ivy.Array(object)
        self.object = object

    # Instance Methoods #
    # -------------------#

    # Add argmax #
    def argmax(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):

        return np_frontend.argmax(
            self.object,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, shape, order="C"):
        return np_frontend.reshape(self.object, shape)

    def transpose(self, /, axes=None):
        return np_frontend.transpose(self.object, axes=axes)

    def add(
        self,
        value,
    ):
        return np_frontend.add(
            self.object,
            value,
        )

    def squeeze(self, axis=None):
        return np_frontend.squeeze(self.object, axis)

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.all(self.object, axis, out, keepdims, where=where)

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.any(self.object, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self.object, axis, kind, order)
