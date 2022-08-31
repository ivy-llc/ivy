# global

# local
import ivy
import ivy.functional.frontends.torch as ivy_frontend


class Tensor:
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


def reshape(self, shape):
    return ivy_frontend.reshape(self, shape)


def add(self, other, *, alpha=1, out=None):
    return ivy_frontend.add(self, other * alpha, out=out)
