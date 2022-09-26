# global

# local
from tkinter.messagebox import NO
import ivy
import ivy.functional.frontends.numpy as np_frontend


class ndarray:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

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
            self.data, 
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, newshape, copy=None):
        return np_frontend.reshape(self.data, newshape, copy=copy)

    def add(
        self,
        other,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
    ):
        return np_frontend.add(
            self.data,
            other,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
