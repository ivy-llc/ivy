# For Review
# global
import abc
from typing import Optional, Union, Tuple, List, Iterable, Sequence
from numbers import Number

# local
import ivy
from ivy import handle_view

# ToDo: implement all methods here as public instance methods


class _ArrayWithManipulation(abc.ABC):
    def view(
        self: ivy.Array,
        /,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, Sequence[int]]] = None,
    ) -> ivy.Array:
        if shape:
            return self.reshape(shape)
        return self.reshape(self.shape)

    def concat(
        self: ivy.Array,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray], ...],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        /,
        *,
        axis: int = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.concat. This method simply wraps the
        function, and so the docstring for ivy.concat also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array to join with other arrays ``xs``.
        xs
            The other arrays to join with. The arrays must
            have the same shape, except in the dimension
            specified by axis.
        axis
            axis along which the arrays will be joined. If axis is None, arrays
            must be flattened before concatenation. If axis is negative, axis on
            which to join arrays is determined by counting from the top. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output array containing the concatenated values.
        """
        return ivy.concat([self._data] + xs, axis=axis, out=out)

    @handle_view
    def expand_dims(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = None,
        axis: Union[int, Sequence[int]] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.expand_dims. This method simply wraps
        the function, and so the docstring for ivy.expand_dims also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            position in the expanded array where a new axis (dimension) of size one
            will be added. If array ``self`` has the rank of ``N``, the ``axis`` needs
            to be between ``[-N-1, N]``. Default: ``0``.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elements of ``self``, but with its dimension added
            by one in a given ``axis``.

        Examples
        --------
        >>> x = ivy.array([-4.7, -2.3, 0.7]) #x.shape->(3,)
        >>> y = x.expand_dims() #y.shape->(1, 3)
        >>> print(y)
        ivy.array([[-4.7, -2.3,  0.7]])
        """
        return ivy.expand_dims(self._data, copy=copy, axis=axis, out=out)

    @handle_view
    def flip(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.flip. This method simply wraps the
        function, and so the docstring for ivy.flip also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis (or axes) along which to flip. If axis is None, all
            input array axes are flipped. If axis is negative, axis
            is counted from the last dimension. If provided more than
            one axis, only the specified axes. Default: None.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an output array having the same data type and
            shape as``self`` and whose elements, relative
            to ``self``, are reordered.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = x.flip()
        >>> print(y)
        ivy.array([3, 2, 1])

        >>> x = ivy.array([4, 5, 6])
        >>> y = x.flip(axis=0)
        >>> print(y)
        ivy.array([6, 5, 4])
        """
        return ivy.flip(self._data, copy=copy, axis=axis, out=out)

    @handle_view
    def permute_dims(
        self: ivy.Array,
        /,
        axes: Tuple[int, ...],
        *,
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.permute_dims. This method simply wraps
        the function, and so the docstring for ivy.permute_dims also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axes
            tuple containing a permutation of (0, 1, ..., N-1) where N is
            the number of axes (dimensions) of x.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the axes permutation. The returned array
            must have the same data type as x.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
        >>> y = x.permute_dims(axes=(1, 0))
        >>> print(y)
        ivy.array([[1, 4],
                   [2, 5],
                   [3, 6]])

        >>> x = ivy.zeros((2, 3))
        >>> y = x.permute_dims(axes=(1, 0))
        >>> print(y)
        ivy.array([[0., 0.],
                   [0., 0.],
                   [0., 0.]])
        """
        return ivy.permute_dims(self._data, axes, copy=copy, out=out)

    @handle_view
    def reshape(
        self: ivy.Array,
        /,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        copy: Optional[bool] = None,
        order: str = "C",
        allowzero: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.reshape. This method simply wraps the
        function, and so the docstring for ivy.reshape also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array.
        shape
            The new shape should be compatible with the original shape.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        order
            Read the elements of the input array using this index order,
            and place the elements into the reshaped array using this index order.
            ‘C’ means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis index
            changing slowest.
            ‘F’ means to read / write the elements using Fortran-like index order, with
            the first index changing fastest, and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout
            of the underlying array, and only refer to the order of indexing.
            Default order is 'C'
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an output array having the same data type as ``self``
            and  elements as ``self``.

        Examples
        --------
        >>> x = ivy.array([[0., 1., 2.],[3., 4., 5.]])
        >>> y = x.reshape((3,2))
        >>> print(y)
        ivy.array([[0., 1.],
                   [2., 3.],
                   [4., 5.]])

        >>> x = ivy.array([[0., 1., 2.],[3., 4., 5.]])
        >>> y = x.reshape((3,2), order='F')
        >>> print(y)
        ivy.array([[0., 4.],
                   [3., 2.],
                   [1., 5.]])
        """
        return ivy.reshape(
            self._data, shape, copy=copy, allowzero=allowzero, out=out, order=order
        )

    def roll(
        self: ivy.Array,
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.roll. This method simply wraps the
        function, and so the docstring for ivy.roll also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positive, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an output array having the same data type as ``self`` and whose elements,
            relative to ``self``, are shifted.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.roll(1)
        >>> print(y)
        ivy.array([2., 0., 1.])

        >>> x = ivy.array([[0., 1., 2.],
        ...                [3., 4., 5.]])
        >>> y = x.roll(2, axis=-1)
        >>> print(y)
        ivy.array([[1., 2., 0.],
                    [4., 5., 3.]])
        """
        return ivy.roll(self._data, shift=shift, axis=axis, out=out)

    @handle_view
    def squeeze(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]],
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.squeeze. This method simply wraps the
        function, and so the docstring for ivy.squeeze also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis (or axes) to squeeze. If a specified axis has a size greater than one,
            a ValueError is. If None, then all squeezable axes are squeezed.
            Default: ``None``.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an output array having the same data type and elements as x.


        Examples
        --------
        >>> x = ivy.array([[[0.],[ 1.]]])
        >>> y = x.squeeze(2)
        >>> print(y)
        ivy.array([[0., 1.]])
        """
        return ivy.squeeze(self._data, axis=axis, copy=copy, out=out)

    def stack(
        self: ivy.Array,
        /,
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        *,
        axis: int = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.stack. This method simply wraps the
        function, and so the docstring for ivy.stack also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Array to join with other ``arrays``.
        arrays
            Other arrays to join with. Each array must have the same shape.
        axis
            axis along which the arrays will be joined. More details can be found in
            the ``ivy.stack`` documentation.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            output array made by joining the input arrays along the specified axis.

        Examples
        --------
        >>> x = ivy.array([1, 2])
        >>> y = ivy.array([5, 6])
        >>> print(x.stack(y, axis=1))
        ivy.array([[1, 5],
                [2, 6]])

        >>> x.stack([y],axis=0)
        ivy.array([[[1, 2]],
                [[5, 6]]])
        """
        if not isinstance(arrays, (tuple, list)):
            arrays = [arrays]
        if isinstance(arrays, tuple):
            x = (self._data,) + arrays
        else:
            x = [self._data] + arrays
        return ivy.stack(x, axis=axis, out=out)

    def clip(
        self: ivy.Array,
        x_min: Union[Number, ivy.Array, ivy.NativeArray],
        x_max: Union[Number, ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.clip. This method simply wraps the
        function, and so the docstring for ivy.clip also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elements of self, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> y = x.clip(1., 5.)
        >>> print(y)
        ivy.array([1., 1., 2., 3., 4., 5., 5., 5., 5., 5.])
        """
        return ivy.clip(self._data, x_min, x_max, out=out)

    def constant_pad(
        self: ivy.Array,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        value: Number = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.constant_pad. This method simply wraps
        the function, and so the docstring for ivy.constant_pad also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array to pad.
        pad_width
            Number of values padded to the edges of each axis.
            Specified as ((before_1, after_1), … (before_N, after_N)), where N
            is number of axes of x.
        value
            The constant value to pad the array with.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            Padded array of rank equal to x with shape increased according
            to pad_width.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> y = x.constant_pad(pad_width = [[2, 3]])
        >>> print(y)
        ivy.array([0., 0., 1., 2., 3., 0., 0., 0.])
        """
        return ivy.constant_pad(self._data, pad_width=pad_width, value=value, out=out)

    def repeat(
        self: ivy.Array,
        /,
        repeats: Union[int, Iterable[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.repeat. This method simply wraps the
        function, and so the docstring for ivy.repeat also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        repeats
            The number of repetitions for each element. repeats is broadcast to
            fit the shape of the given axis.
        axis
            The axis along which to repeat values. By default, use the flattened
            input array, and return a flat output array.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The repeated output array.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y= x.repeat(2)
        >>> print(y)
        ivy.array([0., 0., 1., 1., 2., 2.])
        """
        return ivy.repeat(self._data, repeats=repeats, axis=axis, out=out)

    @handle_view
    def split(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = None,
        num_or_size_splits: Optional[
            Union[int, Sequence[int], ivy.Array, ivy.NativeArray]
        ] = None,
        axis: int = 0,
        with_remainder: bool = False,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.split. This method simply wraps the
        function, and so the docstring for ivy.split also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            array to be divided into sub-arrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers or
            1-D array. Default is to divide into as many 1-dimensional arrays
            as the axis dimension.
        axis
            The axis along which to split, default is ``0``.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is ``False``.

        Returns
        -------
            A list of sub-arrays.

        Examples
        --------
        >>> x = ivy.array([4, 6, 5, 3])
        >>> y = x.split()
        >>> print(y)
        [ivy.array([4]),ivy.array([6]),ivy.array([5]),ivy.array([3])]
        """
        return ivy.split(
            self._data,
            copy=copy,
            num_or_size_splits=num_or_size_splits,
            axis=axis,
            with_remainder=with_remainder,
        )

    @handle_view
    def swapaxes(
        self: ivy.Array,
        axis0: int,
        axis1: int,
        /,
        *,
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.swap_axes. This method simply wraps the
        function, and so the docstring for ivy.split also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis0
            First axis to be swapped.
        axis1
            Second axis to be swapped.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            x with its axes permuted.

        Examples
        --------
        Using :class:`ivy.Array` instance method:

        >>> x = ivy.array([[0., 1., 2.]])
        >>> y = x.swapaxes(0, 1)
        >>> print(y)
        ivy.array([[0.],
                   [1.],
                   [2.]])

        >>> x = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> y = x.swapaxes(0, 2)
        >>> print(y)
        ivy.array([[[0, 4],
                    [2, 6]],
                   [[1, 5],
                    [3, 7]]])
        """
        return ivy.swapaxes(self._data, axis0, axis1, copy=copy, out=out)

    def tile(
        self: ivy.Array,
        /,
        repeats: Iterable[int],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tile. This method simply wraps the
        function, and so the docstring for ivy.tile also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array.
        repeats
            The number of repetitions of x along each axis.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The tiled output array.

        Examples
        --------
        >>> x = ivy.array([[0], [1], [2]])
        >>> y = x.tile((3,2))
        >>> print(y)
        ivy.array([[0,0],
                   [1,1],
                   [2,2],
                   [0,0],
                   [1,1],
                   [2,2],
                   [0,0],
                   [1,1],
                   [2,2]])
        """
        return ivy.tile(self._data, repeats=repeats, out=out)

    @handle_view
    def unstack(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = None,
        axis: int = 0,
        keepdims: bool = False,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.unstack. This method simply wraps the
        function, and so the docstring for ivy.unstack also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array to unstack.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            Axis for which to unpack the array.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is ``False``.

        Returns
        -------
        ret
            List of arrays, unpacked along specified dimensions.

        Examples
        --------
        >>> x = ivy.array([[1, 2], [3, 4]])
        >>> y = x.unstack(axis=0)
        >>> print(y)
        [ivy.array([1, 2]), ivy.array([3, 4])]

        >>> x = ivy.array([[1, 2], [3, 4]])
        >>> y = x.unstack(axis=1, keepdims=True)
        >>> print(y)
        [ivy.array([[1],
                [3]]), ivy.array([[2],
                [4]])]
        """
        return ivy.unstack(self._data, copy=copy, axis=axis, keepdims=keepdims)

    def zero_pad(
        self: ivy.Array,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.zero_pad. This method simply wraps the
        function, and so the docstring for ivy.zero_pad also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input array to pad.
        pad_width
            Number of values padded to the edges of each axis. Specified as
            ((before_1, after_1), … (before_N, after_N)),
            where N is number of axes of x.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Padded array of rank equal to x with shape increased according to pad_width.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([1., 2., 3.,4, 5, 6])
        >>> y = x.zero_pad(pad_width = [[2, 3]])
        >>> print(y)
        ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.])
        """
        return ivy.zero_pad(self._data, pad_width=pad_width, out=out)
