# global
import abc
from typing import Optional

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithActivations(abc.ABC):
    def softplus(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softplus. This method simply wraps the
        function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the softplus activation of each element in ``self``.

        """
        return ivy.softplus(self._data, out=out)

    def tanh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tanh. This method simply wraps the
        function, and so the docstring for ivy.tanh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle. Should have a
            real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic tangent of each element in ``self``. The
            returned array must have a real-valued floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.tanh()
        >>> print(y)
        ivy.array([0., 0.762, 0.964])
        """
        return ivy.tanh(self._data, out=out)
