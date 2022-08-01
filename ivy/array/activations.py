# global
import abc
from typing import Optional

# local
import ivy


# ToDo: implement all methods here as public instance methods


class ArrayWithActivations(abc.ABC):
    def relu(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.relu. This method simply wraps the
        function, and so the docstring for ivy.relu also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-1., 0., 1.])
        >>> y = x.relu()
        >>> print(y)
        ivy.array([0., 0., 1.])
        """
        return ivy.relu(self._data, out=out)

    def leaky_relu(
        self: ivy.Array,
        alpha: Optional[float] = 0.2,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.leaky_relu. This method simply wraps
        the function, and so the docstring for ivy.leaky_relu also applies to this
        method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.leaky_relu()
        >>> print(y)
        ivy.array([ 0.39, -0.17])
        """
        return ivy.leaky_relu(self._data, alpha, out=out)

    def gelu(
        self: ivy.Array,
        approximate: Optional[bool] = True,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gelu. This method simply wraps the
        function, and so the docstring for ivy.gelu also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0.3, -0.1])
        >>> y = x.gelu()
        >>> print(y)
        ivy.array([ 0.185, -0.046])
        """
        return ivy.gelu(self._data, approximate, out=out)

    def tanh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tanh. This method simply wraps the
        function, and so the docstring for ivy.tanh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic tangent of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.tanh()
        >>> print(y)
        ivy.array([0., 0.762, 0.964])
        """
        return ivy.tanh(self._data, out=out)

    def sigmoid(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sigmoid. This method simply wraps the
        function, and so the docstring for ivy.sigmoid also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-1., 1., 2.])
        >>> y = x.sigmoid()
        >>> print(y)
        ivy.array([0.269, 0.731, 0.881])
        """
        return ivy.sigmoid(self._data, out=out)

    def softmax(
        self: ivy.Array, axis: Optional[int] = None, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softmax. This method simply wraps the
        function, and so the docstring for ivy.softmax also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1.0, 0, 1.0])
        >>> y = x.softmax()
        >>> print(y)
        ivy.array([0.422, 0.155, 0.422])
        """
        return ivy.softmax(self._data, axis, out=out)

    def softplus(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softplus. This method simply wraps the
        function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softplus()
        >>> print(y)
        ivy.array([0.535, 0.42 ])
        """
        return ivy.softplus(self._data, out=out)
