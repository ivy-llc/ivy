# global
import abc
from typing import Optional

# local
import ivy


class ArrayWithElementWiseExtensions(abc.ABC):
    def sinc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the sinc of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
        >>> y = x.sinc()
        >>> print(y)
        ivy.array([0.637,-0.212,0.127,-0.0909])
        """
        return ivy.sinc(self._data, out=out)

    def lcm(
        self: ivy.Array, x2: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array.
        x2
            second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            an array that includes the element-wise least common multiples
            of 'self' and x2

        Examples
        --------
        >>> x1=ivy.array([2, 3, 4])
        >>> x2=ivy.array([5, 8, 15])
        >>> x1.lcm(x2)
        ivy.array([10, 21, 60])
        """
        return ivy.lcm(self, x2, out=out)

    def fmod(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.fmod. This method simply
        wraps the function, and so the docstring for ivy.fmod also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
        x1
            First input array.
        x2
            Second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.array([2, 3, 4])
        >>> x2 = ivy.array([1, 5, 2])
        >>> x1.fmod(x2)
        ivy.array([ 0,  3,  0])

        >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
        >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
        >>> x1.fmod(x2)
        ivy.array([ nan,  nan,  nan])
        """
        return ivy.fmod(self._data, x2, out=out)

    def fmax(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.fmax. This method simply
        wraps the function, and so the docstring for ivy.fmax also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
        x1
            First input array.
        x2
            Second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.array([2, 3, 4])
        >>> x2 = ivy.array([1, 5, 2])
        >>> ivy.fmax(x1, x2)
        ivy.array([ 2.,  5.,  4.])

        >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
        >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
        >>> x1.fmax(x2)
        ivy.array([ 0,  0,  nan])
        """
        return ivy.fmax(self._data, x2, out=out)
