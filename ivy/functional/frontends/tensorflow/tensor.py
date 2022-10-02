# global

# local
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend


class Tensor:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

    # Instance Methods #
    # -------------------#

    def Reshape(self, shape, name="Reshape"):
        return tf_frontend.Reshape(tensor=self.data, shape=shape, name=name)

    def add(self, y, name="add"):
        return tf_frontend.add(self.data, y, name)

    def get_shape(self):
        return tf_frontend.Shape(input=self.data)
