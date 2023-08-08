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
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.moveaxis. This method simply wraps the
        function, and so the docstring for ivy.unstack also applies to this method with
        minimal changes.

        Parameters
        ----------
        a
            The array whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
        return ivy.moveaxis(self._data, source, destination, copy=copy, out=out)

    def heaviside(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.heaviside. This method simply wraps the
        function, and so the docstring for ivy.heaviside also applies to this method
        with minimal changes.

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
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.flipud. This method simply wraps the
        function, and so the docstring for ivy.flipud also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The array to be flipped.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
        return ivy.flipud(self._data, copy=copy, out=out)

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
        ivy.Array instance method variant of ivy.vstack. This method simply wraps the
        function, and so the docstring for ivy.vstack also applies to this method with
        minimal changes.

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
        ivy.Array instance method variant of ivy.hstack. This method simply wraps the
        function, and so the docstring for ivy.hstack also applies to this method with
        minimal changes.

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
        copy: bool = None,
        k: int = 1,
        axes: Tuple[int, int] = (0, 1),
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.rot90. This method simply wraps the
        function, and so the docstring for ivy.rot90 also applies to this method with
        minimal changes.

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
        return ivy.rot90(self._data, copy=copy, k=k, axes=axes, out=out)

    def top_k(
        self: ivy.Array,
        k: int,
        /,
        *,
        axis: int = -1,
        largest: bool = True,
        sorted: bool = True,
        out: Optional[tuple] = None,
    ) -> Tuple[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.top_k. This method simply wraps the
        function, and so the docstring for ivy.top_k also applies to this method with
        minimal changes.

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
        sorted
            If sorted is set to True we return the elements in sorted order.
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
        return ivy.top_k(self, k, axis=axis, largest=largest, sorted=sorted, out=out)

    @handle_view
    def fliplr(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fliplr. This method simply wraps the
        function, and so the docstring for ivy.fliplr also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The array to be flipped. Must be at least 2-D.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
        return ivy.fliplr(self._data, copy=copy, out=out)

    def i0(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.i0. This method simply wraps the
        function, and so the docstring for ivy.i0 also applies to this method with
        minimal changes.

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
        copy: Optional[bool] = None,
        start_dim: int = 0,
        end_dim: int = -1,
        order: str = "C",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.flatten. This method simply wraps the
        function, and so the docstring for ivy.flatten also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array to flatten.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
        return ivy.flatten(
            self._data,
            copy=copy,
            start_dim=start_dim,
            end_dim=end_dim,
            order=order,
            out=out,
        )

    def pad(
        self: ivy.Array,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Union[
            Literal[
                "constant",
                "dilated",
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
        stat_length: Union[Iterable[Tuple[int]], int] = 1,
        constant_values: Union[Iterable[Tuple[Number]], Number] = 0,
        end_values: Union[Iterable[Tuple[Number]], Number] = 0,
        reflect_type: Literal["even", "odd"] = "even",
        out: Optional[ivy.Array] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.pad.

        This method simply wraps the function, and so the docstring for
        ivy.pad also applies to this method with minimal changes.
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
        indices_or_sections: Union[int, Sequence[int], ivy.Array],
        /,
        *,
        copy: Optional[bool] = None,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.vsplit. This method simply wraps the
        function, and so the docstring for ivy.vsplit also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        indices_or_sections
            If indices_or_sections is an integer n, the array is split into n
            equal sections, provided that n must be a divisor of the split axis.
            If indices_or_sections is a sequence of ints or 1-D array,
            then input is split at each of the indices.

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
        return ivy.vsplit(self._data, indices_or_sections, copy=copy)

    @handle_view
    def dsplit(
        self: ivy.Array,
        indices_or_sections: Union[int, Sequence[int], ivy.Array],
        /,
        *,
        copy: Optional[bool] = None,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.dsplit. This method simply wraps the
        function, and so the docstring for ivy.dsplit also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        indices_or_sections
            If indices_or_sections is an integer n, the array is split into n
            equal sections, provided that n must be a divisor of the split axis.
            If indices_or_sections is a sequence of ints or 1-D array,
            then input is split at each of the indices.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.

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
        return ivy.dsplit(self._data, indices_or_sections, copy=copy)

    @handle_view
    def atleast_1d(
        self: ivy.Array,
        *arys: Union[ivy.Array, bool, Number],
        copy: Optional[bool] = None,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.atleast_1d. This method simply wraps
        the function, and so the docstring for ivy.atleast_1d also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array. Cannot be a scalar input.
        arys
            An arbitrary number of input arrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.

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
        return ivy.atleast_1d(self._data, *arys, copy=copy)

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
        ivy.Array instance method variant of ivy.dstack. This method simply wraps the
        function, and so the docstring for ivy.dstack also applies to this method with
        minimal changes.

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
    def atleast_2d(
        self: ivy.Array,
        *arys: ivy.Array,
        copy: Optional[bool] = None,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.atleast_2d. This method simply wraps
        the function, and so the docstring for ivy.atleast_2d also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array. Cannot be a scalar input.
        arys
            An arbitrary number of input arrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.

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
        return ivy.atleast_2d(self._data, *arys, copy=copy)

    @handle_view
    def atleast_3d(
        self: ivy.Array,
        *arys: Union[ivy.Array, bool, Number],
        copy: Optional[bool] = None,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.atleast_3d. This method simply wraps
        the function, and so the docstring for ivy.atleast_3d also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array. Cannot be a scalar input.
        arys
            An arbitrary number of input arrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.

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
        return ivy.atleast_3d(self._data, *arys, copy=copy)

    def take_along_axis(
        self: ivy.Array,
        indices: ivy.Array,
        axis: int,
        /,
        *,
        mode: str = "fill",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.take_along_axis. This method simply
        wraps the function, and so the docstring for ivy.take_along_axis also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The source array.
        indices
            The indices of the values to extract.
        axis
            The axis over which to select values.
        mode
            One of: 'clip', 'fill', 'drop'. Parameter controlling how out-of-bounds
            indices will be handled.
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
        return ivy.take_along_axis(self._data, indices, axis, mode=mode, out=out)

    @handle_view
    def hsplit(
        self: ivy.Array,
        indices_or_sections: Union[int, Tuple[int, ...]],
        /,
        *,
        copy: Optional[bool] = None,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.hsplit. This method simply wraps the
        function, and so the docstring for ivy.hsplit also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        indices_or_sections
            If indices_or_sections is an integer n, the array is split into n
            equal sections, provided that n must be a divisor of the split axis.
            If indices_or_sections is a sequence of ints or 1-D array,
            then input is split at each of the indices.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.

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
        return ivy.hsplit(self._data, indices_or_sections, copy=copy)

    @handle_view
    def expand(
        self: ivy.Array,
        shape: Union[ivy.Shape, ivy.NativeShape],
        /,
        *,
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Broadcast the input Array following the given shape and the broadcast rule.

        Parameters
        ----------
        self
            Array input.
        shape
            A 1-D Array indicates the shape you want to expand to,
            following the broadcast rule
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Output Array
        """
        return ivy.expand(self._data, shape, copy=copy, out=out)

    def as_strided(
        self: ivy.Array,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        strides: Sequence[int],
        /,
    ) -> ivy.Array:
        """
        Create a copy of the input array with the given shape and strides.

        Parameters
        ----------
        self
            Input Array.
        shape
            The shape of the new array.
        strides
            The strides of the new array (specified in bytes).

        Returns
        -------
        ret
            Output Array
        """
        return ivy.as_strided(self._data, shape, strides)

    @handle_view
    def concat_from_sequence(
        self: ivy.Array,
        /,
        input_sequence: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        *,
        new_axis: int = 0,
        axis: int = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Concatenate a sequence of arrays along a new or an existing axis.

        Parameters
        ----------
        self
            Array input.
        input_sequence
            A sequence of arrays.
        new_axis
            Insert and concatenate on a new axis or not,
            default 0 means do not insert new axis.
            new_axis = 0: concatenate
            new_axis = 1: stack
        axis
            The axis along which the arrays will be concatenated.
        out
            Optional output array, for writing the result to.

        Returns
        -------
        ret
            Output Array
        """
        if new_axis == 0:
            return ivy.concat_from_sequence(
                [self._data] + input_sequence, new_axis=new_axis, axis=axis, out=out
            )
        elif new_axis == 1:
            if not isinstance(input_sequence, (tuple, list)):
                input_sequence = [input_sequence]
            if isinstance(input_sequence, tuple):
                input_sequence = (self._data,) + input_sequence
            else:
                input_sequence = [self._data] + input_sequence
            return ivy.concat_from_sequence(
                input_sequence, new_axis=new_axis, axis=axis, out=out
            )

    @handle_view
    def associative_scan(
        self: ivy.Array,
        fn: Callable,
        /,
        *,
        reverse: bool = False,
        axis: int = 0,
    ) -> ivy.Array:
        """
        Perform an associative scan over the given array.

        Parameters
        ----------
        self
            The array to scan over.
        fn
            The associative function to apply.
        reverse
            Whether to scan in reverse with respect to the given axis.
        axis
            The axis to scan over.

        Returns
        -------
        ret
            The result of the scan.
        """
        return ivy.associative_scan(self._data, fn, reverse=reverse, axis=axis)

    def unique_consecutive(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
    ) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_consecutive.

        This method simply wraps the function, and so the docstring for
        ivy.unique_consecutive also applies to this method with minimal
        changes.
        """
        return ivy.unique_consecutive(self._data, axis=axis)

    def fill_diagonal(
        self: ivy.Array,
        v: Union[int, float],
        /,
        *,
        wrap: bool = False,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fill_diag.

        This method simply wraps the function, and so the docstring for
        ivy.fill_diag also applies to this method with minimal changes.
        """
        return ivy.fill_diagonal(self._data, v, wrap=wrap)

    def unfold(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        mode: Optional[int] = 0,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.unfold. This method simply wraps the
        function, and so the docstring for ivy.unfold also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input tensor to be unfolded
        mode
            indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            unfolded_tensor of shape ``(tensor.shape[mode], -1)``
        """
        return ivy.unfold(self._data, mode, out=out)

    def fold(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        mode: int,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.fold. This method simply wraps the
        function, and so the docstring for ivy.fold also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            unfolded tensor of shape ``(shape[mode], -1)``
        mode
            the mode of the unfolding
        shape
            shape of the original tensor before unfolding
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            folded_tensor of shape `shape`
        """
        return ivy.fold(self._data, mode, shape, out=out)

    def partial_unfold(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        mode: Optional[int] = 0,
        skip_begin: Optional[int] = 1,
        skip_end: Optional[int] = 0,
        ravel_tensors: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.partial_unfold. This method simply
        wraps the function, and so the docstring for ivy.partial_unfold also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            tensor of shape n_samples x n_1 x n_2 x ... x n_i
        mode
            indexing starts at 0, therefore mode is in range(0, tensor.ndim)
        skip_begin
            number of dimensions to leave untouched at the beginning
        skip_end
            number of dimensions to leave untouched at the end
        ravel_tensors
            if True, the unfolded tensors are also flattened
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            partially unfolded tensor
        """
        return ivy.partial_unfold(
            self._data,
            mode=mode,
            skip_begin=skip_begin,
            skip_end=skip_end,
            ravel_tensors=ravel_tensors,
            out=out,
        )

    def partial_fold(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        mode: int,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        skip_begin: Optional[int] = 1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.partial_fold. This method simply wraps
        the function, and so the docstring for ivy.partial_fold also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            a partially unfolded tensor
        mode
            indexing starts at 0, therefore mode is in range(0, tensor.ndim)
        shape
            the shape of the original full tensor (including skipped dimensions)
        skip_begin
            number of dimensions left untouched at the beginning
        out
            optional output array, for writing the result to.

        Returns
        -------
            partially re-folded tensor
        """
        return ivy.partial_fold(self._data, mode, shape, skip_begin, out=out)

    def partial_tensor_to_vec(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        skip_begin: Optional[int] = 1,
        skip_end: Optional[int] = 0,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.partial_tensor_to_vec. This method
        simply wraps the function, and so the docstring for ivy.partial_tensor_to_vec
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            tensor to partially vectorise
        skip_begin
            number of dimensions to leave untouched at the beginning
        skip_end
            number of dimensions to leave untouched at the end
        out
            optional output array, for writing the result to.

        Returns
        -------
            partially vectorised tensor with the
            `skip_begin` first and `skip_end` last dimensions untouched
        """
        return ivy.partial_tensor_to_vec(self._data, skip_begin, skip_end, out=out)

    def partial_vec_to_tensor(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        skip_begin: Optional[int] = 1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.partial_vec_to_tensor. This method
        simply wraps the function, and so the docstring for ivy.partial_vec_to_tensor
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            a partially vectorised tensor
        shape
            the shape of the original full tensor (including skipped dimensions)
        skip_begin
            number of dimensions to leave untouched at the beginning
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            full tensor
        """
        return ivy.partial_vec_to_tensor(self._data, shape, skip_begin, out=out)

    def matricize(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        row_modes: Sequence[int],
        column_modes: Optional[Sequence[int]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.matricize. This method simply wraps the
        function, and so the docstring for ivy.matricize also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            the input tensor
        row_modes
            modes to use as row of the matrix (in the desired order)
        column_modes
            modes to use as column of the matrix, in the desired order
            if None, the modes not in `row_modes` will be used in ascending order
        out
            optional output array, for writing the result to.

        ret
        -------
            ivy.Array : tensor of size (ivy.prod(x.shape[i] for i in row_modes), -1)
        """
        return ivy.matricize(self._data, row_modes, column_modes, out=out)

    def soft_thresholding(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        threshold: Union[float, ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.soft_thresholding. This method simply
        wraps the function, and so the docstring for ivy.soft_thresholding also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input array
        threshold
            float or array with shape tensor.shape
            * If float the threshold is applied to the whole tensor
            * If array, one threshold is applied per elements, 0 values are ignored
        out
            optional output array, for writing the result to.

        Returns
        -------
        ivy.Array
            thresholded tensor on which the operator has been applied
        """
        return ivy.soft_thresholding(self._data, threshold, out=out)
