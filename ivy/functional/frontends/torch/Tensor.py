# global

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend


class Tensor:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

    # Instance Methoods #
    # -------------------#

    def reshape(self, shape):
        return torch_frontend.reshape(self.data, shape)

    def add(self, other, *, alpha=1):
        return torch_frontend.add(self.data, other, alpha=alpha)

    def sin(self, *, out=None):
        return torch_frontend.sin(self.data, out=out)

    def sin_(self):
        self.data = self.sin()
        return self.data


# Tensor (alias)
tensor = Tensor
