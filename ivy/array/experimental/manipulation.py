# global
import abc
from typing import (Optional,
                    Union,
                    Sequence,
                    Tuple,
                    List,
                    Iterable,
                    Callable,
                    Literal,
                    Any)
from numbers import Number

# local
import ivy


class ArrayWithManipulationExperimental(abc.ABC):
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

    def fliplr(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.fliplr. This method simply
        wraps the function, and so the docstring for ivy.fliplr also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The array to be flipped. Must be at least 2-D.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array corresponding to input array with elements
            order reversed along axis 1.

        Examples
        --------
        >>> m = ivy.diag([1, 2, 3])
        >>> m.fliplr()
        ivy.array([[0, 0, 1],
               [0, 2, 0],
               [3, 0, 0]])
        """
        return ivy.fliplr(self._data, out=out)

    def i0(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.i0. This method simply
        wraps the function, and so the docstring for ivy.i0 also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Optional output, for writing the result to.

        Returns
        -------
        ret
            Array with modified Bessel function of the first kind, order 0.

        Examples
        --------
        >>> x = ivy.array([[1, 2, 3]])
        >>> x.i0()
        ivy.array([1.26606588, 2.2795853 , 4.88079259])
        """
        return ivy.i0(self._data, out=out)

    def flatten(
        self: ivy.Array,
        *,
        start_dim: Optional[int] = 0,
        end_dim: Optional[int] = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.flatten. This method simply
        wraps the function, and so the docstring for ivy.flatten also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array to flatten.
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            the flattened array over the specified dimensions.

        Examples
        --------
        >>> x = ivy.array([1,2], [3,4])
        >>> ivy.flatten(x)
        ivy.array([1, 2, 3, 4])

        >>> x = ivy.array(
            [[[[ 5,  5,  0,  6],
            [17, 15, 11, 16],
            [ 6,  3, 13, 12]],

            [[ 6, 18, 10,  4],
            [ 5,  1, 17,  3],
            [14, 14, 18,  6]]],


        [[[12,  0,  1, 13],
            [ 8,  7,  0,  3],
            [19, 12,  6, 17]],

            [[ 4, 15,  6, 15],
            [ 0,  5, 17,  9],
            [ 9,  3,  6, 19]]],


        [[[17, 13, 11, 16],
            [ 4, 18, 17,  4],
            [10, 10,  9,  1]],

            [[19, 17, 13, 10],
            [ 4, 19, 16, 17],
            [ 2, 12,  8, 14]]]]
            )
        >>> ivy.flatten(x, start_dim = 1, end_dim = 2)
        ivy.array(
            [[[ 5,  5,  0,  6],
            [17, 15, 11, 16],
            [ 6,  3, 13, 12],
            [ 6, 18, 10,  4],
            [ 5,  1, 17,  3],
            [14, 14, 18,  6]],

            [[12,  0,  1, 13],
            [ 8,  7,  0,  3],
            [19, 12,  6, 17],
            [ 4, 15,  6, 15],
            [ 0,  5, 17,  9],
            [ 9,  3,  6, 19]],

            [[17, 13, 11, 16],
            [ 4, 18, 17,  4],
            [10, 10,  9,  1],
            [19, 17, 13, 10],
            [ 4, 19, 16, 17],
            [ 2, 12,  8, 14]]]))
        """
        return ivy.flatten(self._data, start_dim=start_dim, end_dim=end_dim, out=out)

    def pad(
        self: ivy.Array,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        out: Optional[ivy.Array] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies
        to this method with minimal changes.
        """
        return ivy.pad(
            self._data,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            out=out,
            **kwargs,
        )
