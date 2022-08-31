# global

# local
import ivy
import ivy.functional.frontends.numpy as ivy_frontend


class ndarray:
    def __init__(self, data):
        self._init(data)

    def _init(self, data):
        if ivy.is_ivy_array(data):
            self.data = data.data
        else:
            assert ivy.is_native_array(data)
            self.data = data

    # Instance Methoods #
    # -------------------#

    def reshape(self, newshape, order="C"):
        return ivy_frontend.reshape(self, newshape, order)

    def add(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
    ):
        return ivy_frontend.add(
            self,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
