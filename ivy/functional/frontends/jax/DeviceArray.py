# local
import ivy
import ivy.functional.frontends.jax as ivy_frontend


class DeviceArray:
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
    def reshape(self, new_sizes, dimensions=None):
        return ivy_frontend.reshape(self.data, new_sizes, dimensions)

    def add(self, other):
        return ivy_frontend.add(self.data, other)
