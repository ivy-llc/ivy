# global
import ivy


# local
import ivy.functional.frontends.jax as ivy_frontend
import ivy.functional.frontends.frontend_array as ivy_frontend_array


class DeviceArray(ivy_frontend_array.frontend_array):
    def __init__(self, data):
        ivy.set_current_backend("jax")
        self._init(data)
        ivy.unset_backend()

    @property
    def T(self):
        return DeviceArray(ivy_frontend.transpose(self))

    # ToDo: Implement these properties
    @property
    def at(self, idx):
        return None

    @property
    def imag(self, val):
        return None

    @property
    def real(self, val):
        return None

    # Instance Methods
    # ---------------#

    def reshape(self, new_sizes, dimensions=None):
        return ivy_frontend.reshape(self, new_sizes, dimensions)

    def add(self, other):
        return ivy_frontend.add(self, other)
