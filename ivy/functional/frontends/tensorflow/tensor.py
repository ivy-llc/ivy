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
        return tf_frontend.add(self.data, y, name=name)

    def __eq__(self, other):
        return tf_frontend.Equal(
            x=self.data, y=other.data, incompatible_shape_error=False
        )

    def __floordiv__(self, y, name="floordiv"):
        return y.__rfloordiv__(self.data)

    def __ge__(self, y, name="ge"):
        return tf_frontend.GreaterEqual(x=self.data, y=y.data, name=name)

    def __gt__(self, y, name="gt"):
        return tf_frontend.Greater(x=self.data, y=y.data, name=name)

    def __le__(self, y, name="le"):
        return tf_frontend.LessEqual(x=self.data, y=y.data, name=name)

    def __lt__(self, y, name="lt"):
        return tf_frontend.Less(x=self.data, y=y.data, name=name)

    def __ne__(self, other):
        return tf_frontend.NotEqual(
            x=self.data, y=other.data, incompatible_shape_error=False
        )

    def __sub__(self, y, name="sub"):
        return y.__rsub__(self.data)

    def __radd__(self, x, name="radd"):
        return tf_frontend.add(x, self.data, name=name)

    def __rfloordiv__(self, x, name="rfloordiv"):
        return tf_frontend.FloorDiv(x=x, y=self.data, name=name)

    def __rsub__(self, x, name="rsub"):
        return tf_frontend.subtract(x, self.data, name=name)
