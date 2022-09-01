# global

# local
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend


class Tensor:
    def __init__(self, data):
        if ivy.is_ivy_array(data):
            self.data = data.data
        else:
            assert ivy.is_native_array(data)
            self.data = data

    # Instance Methoods #
    # -------------------#

    def reshape(self, shape, name=None):
        return tf_frontend.reshape(self, shape, name)

    def add(self, y, name=None):
        return tf_frontend.add(self, y, name)
