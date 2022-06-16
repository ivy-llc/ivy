# global
import abc
from typing import Optional, Union

# local
import ivy
# ToDo: implement all methods here as public instance methods

class ArrayWithActivations(abc.ABC):
    def relu(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.relu(self, out=out)

    def leaky_relu(
            self: ivy.Array,
            alpha: Optional[float] = 0.2,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.leaky_relu(self,alpha, out=out)

    def gelu(
            self: ivy.Array,
            approximate: Optional[bool] = True,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.gelu(self, approximate, out=out)

    def tanh(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.tanh(self, out=out)

    def sigmoid(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.sigmoid(self, out=out)

    def softmax(self: ivy.Array,
                axis: Optional[int] = None,
                out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.softmax(self,axis, out=out)
