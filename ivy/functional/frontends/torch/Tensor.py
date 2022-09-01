# global

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend


class Tensor:
    def __init__(self, data):
        if ivy.is_ivy_array(data):
            self.data = data.data
        else:
            assert ivy.is_native_array(data)
            self.data = data

    # Instance Methoods #
    # -------------------#

    def reshape(self, shape):
        return torch_frontend.reshape(self, shape)

    def add(self, other, *, alpha=1, out=None):
        return torch_frontend.add(self, other * alpha, out=out)
