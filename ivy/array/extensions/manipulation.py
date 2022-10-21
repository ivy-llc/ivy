# global
import abc
from typing import Optional, Union, Sequence

# local
import ivy


class ArrayWithManipulationExtensions(abc.ABC):
    def moveaxis(
        self: ivy.Array,
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.moveaxis. This method simply
        wraps the function, and so the docstring for ivy.unstack also applies to
        this method with minimal changes.

        Parameters
        ----------
        a
            The array whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with moved axes. This array is a view of the input array.

        Examples
        --------
        >>> x = ivy.zeros((3, 4, 5))
        >>> x.moveaxis(0, -1).shape
        (4, 5, 3)
        >>> x.moveaxis(-1, 0).shape
        (5, 3, 4)
        """
        return ivy.moveaxis(self._data, source, destination, out=out)

    def heaviside(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.heaviside. This method simply
        wraps the function, and so the docstring for ivy.heaviside also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        x2
            values to use where x1 is zero.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            output array with element-wise Heaviside step function of x1.
            This is a scalar if both x1 and x2 are scalars.

        Examples
        --------
        >>> x1 = ivy.array([-1.5, 0, 2.0])
        >>> x2 = ivy.array([0.5])
        >>> ivy.heaviside(x1, x2)
        ivy.array([0.0000, 0.5000, 1.0000])

        >>> x1 = ivy.array([-1.5, 0, 2.0])
        >>> x2 = ivy.array([1.2, -2.0, 3.5])
        >>> ivy.heaviside(x1, x2)
        ivy.array([0., -2., 1.])
        """
        return ivy.heaviside(self._data, x2, out=out)

    def flipud(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.flipud. This method simply
        wraps the function, and so the docstring for ivy.flipud also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The array to be flipped.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array corresponding to input array with elements
            order reversed along axis 0.

        Examples
        --------
        >>> m = ivy.diag([1, 2, 3])
        >>> m.flipud()
        ivy.array([[ 0.,  0.,  3.],
            [ 0.,  2.,  0.],
            [ 1.,  0.,  0.]])
        """
        return ivy.flipud(self._data, out=out)
