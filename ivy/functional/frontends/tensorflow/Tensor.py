# global

# local
import ivy
import ivy.functional.frontends.tensorflow as ivy_frontend


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

    def reshape(self, shape, name=None):
        return ivy_frontend.reshape(self, shape, name)

    def add(self, y, name=None):
        return ivy_frontend.add(self, y, name)
