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

    def get_shape(self):
        return tf_frontend.Shape(input=self.data)

    def __add__(self, y, name="add"):
        return y.__radd__(self.data)

    def __div__(self, x, name="div"):
        return tf_frontend.divide(x, self.data, name=name)

    def __and__(self, y, name="and"):
        return y.__rand__(self.data)

    def __bool__(self, name="bool"):
        if isinstance(self.data, int):
            return self.data != 0

        temp = ivy.squeeze(ivy.asarray(self.data), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ivy.exceptions.IvyError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __eq__(self, other):
        return tf_frontend.Equal(x=self.data, y=other, incompatible_shape_error=False)

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

    def __mul__(self, x, name="mul"):
        return tf_frontend.multiply(x, self.data, name=name)

    def __ne__(self, other):
        return tf_frontend.NotEqual(
            x=self.data, y=other.data, incompatible_shape_error=False
        )

    def __neg__(self, name="neg"):
        return tf_frontend.Neg(x=self.data, name=name)

    __nonzero__ = __bool__

    def __or__(self, y, name="or"):
        return y.__ror__(self.data)

    def __radd__(self, x, name="radd"):
        return tf_frontend.add(x, self.data, name=name)

    def __rand__(self, x, name="rand"):
        return tf_frontend.logical_and(x, self.data, name=name)

    def __rfloordiv__(self, x, name="rfloordiv"):
        return tf_frontend.FloorDiv(x=x, y=self.data, name=name)

    def __ror__(self, x, name="ror"):
        return tf_frontend.LogicalOr(x=x, y=self.data, name=name)

    def __rsub__(self, x, name="rsub"):
        return tf_frontend.subtract(x, self.data, name=name)

    def __rtruediv__(self, x, name="rtruediv"):
        return tf_frontend.divide(x, self.data, name=name)

    def __sub__(self, y, name="sub"):
        return y.__rsub__(self.data)

    def __truediv__(self, y, name="truediv"):
        return y.__rtruediv__(self.data)
