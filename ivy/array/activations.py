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
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([-1., 0., 1.])
        >>> y = ivy.relu(x)
        >>> print(y)
        ivy.array([0., 0., 1.])
        """
        return ivy.relu(self._data, out=out)

    def leaky_relu(
            self: ivy.Array,
            alpha: Optional[float] = 0.2,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([0.39, -0.85])
        >>> y = ivy.leaky_relu(x)
        >>> print(y)
        ivy.array([ 0.39, -0.17])
        """
        return ivy.leaky_relu(self._data, alpha, out=out)

    def gelu(
            self: ivy.Array,
            approximate: Optional[bool] = True,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([0.3, -0.1])
        >>> y = ivy.gelu(x)
        >>> print(y)
        ivy.array([ 0.185, -0.046])
        """
        return ivy.gelu(self._data, approximate, out=out)

    def tanh(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([0.55 , -0.55])
        >>> y = ivy.tanh(x)
        >>> print(y)
        ivy.array([ 0.501, -0.501])
        """
        return ivy.tanh(self._data, out=out)

    def sigmoid(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([-1., 1., 2.])
        >>> y = ivy.sigmoid(x)
        >>> print(y)
        ivy.array([0.269, 0.731, 0.881])
        """
        return ivy.sigmoid(self._data, out=out)

    def softmax(self: ivy.Array,
                axis: Optional[int] = None,
                out: Optional[ivy.Array] = None
                ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([1.0, 0, 1.0])
        >>> y = x.softmax()
        >>> print(y)
        ivy.array([0.422, 0.155, 0.422])
        """
        return ivy.softmax(self._data, axis, out=out)

    def softplus(self: ivy.Array,
                out: Optional[ivy.Array] = None
                ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Array` instance method:
        >>> import ivy
        >>> ivy.set_backend("numpy")
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = ivy.softplus(x)
        >>> print(y)
        ivy.array([0.535, 0.42 ])
        """
        return ivy.softplus(self._data, out=out)

