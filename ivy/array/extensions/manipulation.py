# global
import abc
from typing import Optional, Union, Sequence, Tuple, List

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

    def vstack(
        self: ivy.Array,
        /,
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.vstack. This method simply
        wraps the function, and so the docstring for ivy.vstack also applies
        to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([[1, 2]])
        >>> y = [ivy.array([[5, 6]]), ivy.array([[7, 8]])]
        >>> print(x.vstack(y))
            ivy.array([[1, 2],
                       [5, 6],
                       [7, 8]])
        """
        return ivy.vstack(self.concat(arrays), out=out)

    def hstack(
        self: ivy.Array,
        /,
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hstack. This method simply
        wraps the function, and so the docstring for ivy.hstack also applies
        to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([[1, 2]])
        >>> y = [ivy.array([[5, 6]]), ivy.array([[7, 8]])]
        >>> print(x.vstack(y))
        ivy.array([1, 2, 5, 6, 7, 8])

        """
        return ivy.hstack(self.concat(arrays), out=out)

    def rot90(
        self: ivy.Array,
        /,
        *,
        k: Optional[int] = 1,
        axes: Optional[Tuple[int, int]] = (0, 1),
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.rot90.
        This method simply wraps the function, and so the docstring
        for ivy.rot90 also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of two or more dimensions.
        k
            Number of times the array is rotated by 90 degrees.
        axes
            The array is rotated in the plane defined by the axes. Axes must be
            different.
        out
            Optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            Array with a rotated view of input array.

        Examples
        --------
        >>> m = ivy.array([[1,2], [3,4]])
        >>> m.rot90()
        ivy.array([[2, 4],
               [1, 3]])
        >>> m = ivy.array([[1,2], [3,4]])
        >>> m.rot90(k=2)
        ivy.array([[4, 3],
               [2, 1]])
        >>> m = ivy.array([[[0, 1],\
                            [2, 3]],\
                           [[4, 5],\
                            [6, 7]]])
        >>> m.rot90(k=2, axes=(1,2))
        ivy.array([[[3, 2],
                [1, 0]],

               [[7, 6],
                [5, 4]]])

        """
        return ivy.rot90(self._data, k=k, axes=axes, out=out)

    def top_k(
        self: ivy.Array,
        k: int,
        /,
        *,
        axis: Optional[int] = None,
        largest: Optional[bool] = True,
        out: Optional[tuple] = None,
    ) -> Tuple[ivy.Array, ivy.NativeArray]:
        """ivy.Array instance method variant of ivy.top_k. This method simply
        wraps the function, and so the docstring for ivy.top_k also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            The array to compute top_k for.
        k
            Number of top elements to retun must not exceed the array size.
        axis
            The axis along which we must return the top elements default value is 1.
        largest
            If largest is set to False we return k smallest elements of the array.
        out:
            Optional output tuple, for writing the result to. Must have two arrays,
            with a shape that the returned tuple broadcast to.

        Returns
        -------
        ret
            A named tuple with values and indices of top k elements.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([2., 1., -3., 5., 9., 0., -4])
        >>> y = x.top_k(2)
        >>> print(y)
        top_k(values=ivy.array([9., 5.]), indices=ivy.array([4, 3]))
        """
        return ivy.top_k(self, k, axis=axis, largest=largest, out=out)
