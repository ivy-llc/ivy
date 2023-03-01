# global
import abc
from typing import (
    Optional,
    Union,
    Sequence,
    Tuple,
    List,
    Iterable,
    Callable,
    Literal,
    Any,
)
from numbers import Number

# local
import ivy
from ivy import handle_view


class _ArrayWithManipulationExperimental(abc.ABC):
    @handle_view
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

    @handle_view
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
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        /,
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
        if not isinstance(arrays, (list, tuple)):
            arrays = [arrays]
        if isinstance(arrays, tuple):
            x = (self._data) + arrays
        else:
            x = [self._data] + arrays
        return ivy.vstack(x, out=out)

    def hstack(
        self: ivy.Array,
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        /,
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
        if not isinstance(arrays, (list, tuple)):
            arrays = [arrays]
        if isinstance(arrays, tuple):
            x = (self._data,) + arrays
        else:
            x = [self._data] + arrays
        return ivy.hstack(x, out=out)

    @handle_view
    def rot90(
        self: ivy.Array,
        /,
        *,
        k: int = 1,
        axes: Tuple[int, int] = (0, 1),
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
        largest: bool = True,
        out: Optional[tuple] = None,
    ) -> Tuple[ivy.Array, ivy.NativeArray]:
        """ivy.Array instance method variant of ivy.top_k. This method simply
        wraps the function, and so the docstring for ivy.top_k also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
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

    @handle_view
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

    @handle_view
    def flatten(
        self: ivy.Array,
        *,
        start_dim: int = 0,
        end_dim: int = -1,
        order: str = "C",
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
        order
            Read the elements of the input container using this index order,
            and place the elements into the reshaped array using this index order.
            ‘C’ means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis index
            changing slowest.
            ‘F’ means to read / write the elements using Fortran-like index order, with
            the first index changing fastest, and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout
            of the underlying array, and only refer to the order of indexing.
            Default order is 'C'.
        out
            Optional output, for writing the result to.

        Returns
        -------
        ret
            the flattened array over the specified dimensions.

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> x.flatten()
        ivy.array([1, 2, 3, 4])

        >>> x = ivy.array([[1,2], [3,4]])
        >>> x.flatten(order='F')
        ivy.array([1, 3, 2, 4])

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
        >>> x.flatten(start_dim = 1, end_dim = 2)
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
        mode: Union[
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
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Union[Iterable[Tuple[Number]], Number] = 0,
        end_values: Union[Iterable[Tuple[Number]], Number] = 0,
        reflect_type: Literal["even", "odd"] = "even",
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

    @handle_view
    def vsplit(
        self: ivy.Array,
        indices_or_sections: Union[int, Tuple[int, ...]],
        /,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.vsplit. This method simply
        wraps the function, and so the docstring for ivy.vsplit also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        indices_or_sections
            If indices_or_sections is an integer n, the array is split into n
            equal sections, provided that n must be a divisor of the split axis.
            If indices_or_sections is a tuple of ints, then input is split at each of
            the indices in the tuple.

        Returns
        -------
        ret
            input array split vertically.

        Examples
        --------
        >>> ary = ivy.array(
            [[[0.,  1.],
              [2.,  3.]],
             [[4.,  5.],
              [6.,  7.]]]
            )
        >>> ary.vsplit(2)
        [ivy.array([[[0., 1.], [2., 3.]]]), ivy.array([[[4., 5.], [6., 7.]]])])
        """
        return ivy.vsplit(self._data, indices_or_sections)

    @handle_view
    def dsplit(
        self: ivy.Array,
        indices_or_sections: Union[int, Tuple[int, ...]],
        /,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.dsplit. This method simply
        wraps the function, and so the docstring for ivy.dsplit also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        indices_or_sections
            If indices_or_sections is an integer n, the array is split into n
            equal sections, provided that n must be a divisor of the split axis.
            If indices_or_sections is a tuple of ints, then input is split at each of
            the indices in the tuple.

        Returns
        -------
        ret
            input array split along the 3rd axis.

        Examples
        --------
        >>> ary = ivy.array(
            [[[ 0.,   1.,   2.,   3.],
              [ 4.,   5.,   6.,   7.]],
             [[ 8.,   9.,  10.,  11.],
              [12.,  13.,  14.,  15.]]]
        )
        >>> ary.dsplit(2)
        [ivy.array([[[ 0.,  1.], [ 4.,  5.]], [[ 8.,  9.], [12., 13.]]]),
        ivy.array([[[ 2.,  3.], [ 6.,  7.]], [[10., 11.], [14., 15.]]])]
        """
        return ivy.dsplit(self._data, indices_or_sections)

    @handle_view
    def atleast_1d(
        self: ivy.Array, *arys: Union[ivy.Array, bool, Number]
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.atleast_1d. This method simply
        wraps the function, and so the docstring for ivy.atleast_1d also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array. Cannot be a scalar input.
        arys
            An arbitrary number of input arrays.

        Returns
        -------
        ret
            List of arrays, each with a.ndim >= 1. Copies are made
            only if necessary.

        Examples
        --------
        >>> a1 = ivy.array([[1,2,3]])
        >>> a2 = ivy.array(4)
        >>> a1.atleast_1d(a2,5,6)
        [ivy.array([[1, 2, 3]]), ivy.array([4]), ivy.array([5]), ivy.array([6])]
        """
        return ivy.atleast_1d(self._data, *arys)

    @handle_view
    def dstack(
        self: ivy.Array,
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.dstack. This method simply
        wraps the function, and so the docstring for ivy.dstack also applies
        to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([2, 3, 4])
        >>> x.dstack(y)
        ivy.array([[[1, 2],
                    [2, 3],
                    [3, 4]]])
        """
        if not isinstance(arrays, (list, tuple)):
            arrays = [arrays]
        if isinstance(arrays, tuple):
            x = (self._data,) + arrays
        else:
            x = [self._data] + arrays
        return ivy.dstack(x, out=out)

    @handle_view
    def atleast_2d(self: ivy.Array, *arys: ivy.Array) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.atleast_2d. This method simply
        wraps the function, and so the docstring for ivy.atleast_2d also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array. Cannot be a scalar input.
        arys
            An arbitrary number of input arrays.

        Returns
        -------
        ret
            List of arrays, each with a.ndim >= 2. Copies are made
            only if necessary.

        Examples
        --------
        >>> a1 = ivy.array([[1,2,3]])
        >>> a2 = ivy.array(4)
        >>> a1.atleast_2d(a2,5,6)
        [ivy.array([[1, 2, 3]]), ivy.array([[4]]), ivy.array([[5]]), ivy.array([[6]])]
        """
        return ivy.atleast_2d(self._data, *arys)

    @handle_view
    def atleast_3d(
        self: ivy.Array, *arys: Union[ivy.Array, bool, Number]
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.atleast_3d. This method simply
        wraps the function, and so the docstring for ivy.atleast_3d also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array. Cannot be a scalar input.
        arys
            An arbitrary number of input arrays.

        Returns
        -------
        ret
            List of arrays, each with a.ndim >= 3. Copies are made only if necessary
            and views with three or more dimensions are returned. For example, a 1-D
            array of shape (N,) becomes a view of shape (1, N, 1), and a 2-D array
            of shape (M, N) becomes a view of shape (M, N, 1).

        Examples
        --------
        >>> a1 = ivy.array([[1,2,3]])
        >>> a2 = ivy.array([4,8])
        >>> a1.atleast_3d(a2,5,6)
        [ivy.array([[[1],
                [2],
                [3]]]), ivy.array([[[4],
                [8]]]), ivy.array([[[5]]]), ivy.array([[[6]]])]
        """
        return ivy.atleast_3d(self._data, *arys)

    def take_along_axis(
        self: ivy.Array,
        indices: ivy.Array,
        axis: int,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.take_along_axis. This method simply
        wraps the function, and so the docstring for ivy.take_along_axis also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The source array.
        indices
            The indices of the values to extract.
        axis
            The axis over which to select values.
        out
            Optional output, for writing the result to.

        Returns
        -------
        ret
            The returned array has the same shape as indices.

        Examples
        --------
        >>> arr = ivy.array([[4, 3, 5], [1, 2, 1]])
        >>> indices = ivy.array([[0, 1, 1], [2, 0, 0]])
        >>> y = arr.take_along_axis(indices, 1)
        >>> print(y)
        ivy.array([[4, 3, 3], [1, 1, 1]])
        """
        return ivy.take_along_axis(self._data, indices, axis, out=out)

    @handle_view
    def hsplit(
        self: ivy.Array,
        indices_or_sections: Union[int, Tuple[int, ...]],
        /,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.hsplit. This method simply
        wraps the function, and so the docstring for ivy.hsplit also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        indices_or_sections
            If indices_or_sections is an integer n, the array is split into n
            equal sections, provided that n must be a divisor of the split axis.
            If indices_or_sections is a tuple of ints, then input is split at each of
            the indices in the tuple.

        Returns
        -------
        ret
            list of arrays split horizontally from input array.

        Examples
        --------
        >>> ary = ivy.array(
            [[0.,  1., 2., 3.],
             [4.,  5., 6,  7.],
             [8.,  9., 10., 11.],
             [12., 13., 14., 15.]]
            )
        >>> ary.hsplit(2)
        [ivy.array([[ 0.,  1.],
                    [ 4.,  5.],
                    [ 8.,  9.],
                    [12., 13.]]),
         ivy.array([[ 2.,  3.],
                    [ 6.,  7.],
                    [10., 11.],
                    [14., 15.]]))
        """
        return ivy.hsplit(self._data, indices_or_sections)

    @handle_view
    def expand(
        self: ivy.Array,
        shape: Union[ivy.Shape, ivy.NativeShape],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Broadcast the input Array following the given shape
        and the broadcast rule.

        Parameters
        ----------
        self
            Array input.
        shape
            A 1-D Array indicates the shape you want to expand to,
            following the broadcast rule
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Output Array
        """
        return ivy.expand(self._data, shape, out=out)
