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

    def get_shape(self):
        return tf_frontend.Shape(input=self.data)

    def __add__(self, y, name="add"):
        return tf_frontend.add(self.data, y.data, name)

    def __eq__(self, other):
        return tf_frontend.Equal(
            x=self.data, y=other.data, incompatible_shape_error=False
        )

    def __floordiv__(self, y, name="floordiv"):
        return tf_frontend.FloorDiv(x=self.data, y=y.data, name=name)

    def __ge__(self, y, name="ge"):
        return tf_frontend.GreaterEqual(x=self.data, y=y.data, name=name)

    def __gt__(self, y, name="gt"):
        return tf_frontend.Greater(x=self.data, y=y.data, name=name)

    def __le__(self, y, name="le"):
        return tf_frontend.LessEqual(x=self.data, y=y.data, name=name)

    def __lt__(self, y, name="lt"):
        return tf_frontend.Less(x=self.data, y=y.data, name=name)
