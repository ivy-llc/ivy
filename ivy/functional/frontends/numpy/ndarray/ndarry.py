# global

# local
import ivy.functional.frontends.numpy as ivy_frontend


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
