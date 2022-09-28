# global
import numpy as np

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class ndarray:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

    # Instance Methoods #
    # -------------------#

    # argpartition #
    def argpartition(
        self,
        kth,
        axis=-1,
        kind='introselect',
        order=None,
    ):
        return np.argpartition(self.data, 
            kth=kth,
            axis=axis, 
            kind=kind, 
            order=order)

    def reshape(self, shape, order="C"):
        return np_frontend.reshape(self.data, shape)

    def transpose(self, /, axes=None):
        return np_frontend.transpose(self.data, axes=axes)

    def add(
        self,
        value,
    ):
        return np_frontend.add(
            self.data,
            value,
        )
