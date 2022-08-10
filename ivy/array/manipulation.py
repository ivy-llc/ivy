# For Review
# global
import abc
from typing import Optional, Union, Tuple, List, Iterable, Sequence
from numbers import Number

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithManipulation(abc.ABC):
    def concat(
        self: ivy.Array,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        /,
        *,
        axis: Optional[int] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.concat. This method simply
        wraps the function, and so the docstring for ivy.concat also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input arrays to join. The arrays must have the same shape, except in the
            dimension specified by axis.
        axis
            axis along which the arrays will be joined. If axis is None, arrays
            must be flattened before concatenation. If axis is negative, axis on
            which to join arrays is determined by counting from the top. Default: 0.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output array containing the concatenated values.

        """
        return ivy.concat([self._data] + xs, axis, out=out)

    def expand_dims(
        self: ivy.Array,
        /,
        *,
        axis: Union[int, Tuple[int], List[int]] = 0,
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
        return ivy.expand_dims(self._data, axis, out=out)

    def flip(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.flip. This method simply
        wraps the function, and so the docstring for ivy.flip also applies
        to this method with minimal changes.

        """
        return ivy.flip(self._data, axis, out=out)

    def permute_dims(
        self: ivy.Array,
        /,
        axes: Tuple[int, ...],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.permute_dims. This method simply
        wraps the function, and so the docstring for ivy.permute_dims also applies
        to this method with minimal changes.

        """
        return ivy.permute_dims(self._data, axes, out=out)

    def reshape(
        self: ivy.Array,
        /,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        copy: Optional[bool] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.reshape. This method simply wraps the
        function, and so the docstring for ivy.reshape also applies to this method
        with minimal changes.

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
            and copy otherwise. Default: None.
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
        >>> y = x.reshape((2,3))
        >>> print(y)
        ivy.array([[0., 1., 2.],
                   [3., 4., 5.]])

        """
        return ivy.reshape(self._data, shape, copy=copy, out=out)

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
        function, and so the docstring for ivy.roll also applies to this method
        with minimal changes.

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
        """
        return ivy.roll(self._data, shift=shift, axis=axis, out=out)

    def squeeze(
        self: ivy.Array,
        /,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.squeeze. This method simply wraps
        the function, and so the docstring for ivy.squeeze also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([[[0.],[ 1.]]])
        >>> y = x.squeeze(2)
        >>> print(y)
        ivy.array([[0., 1.]])
        """
        return ivy.squeeze(self._data, axis=axis, out=out)

    def stack(
        self: ivy.Array,
        /,
        arrays: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        *,
        axis: Optional[int] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.stack. This method simply
        wraps the function, and so the docstring for ivy.stack also applies
        to this method with minimal changes.
        """
        return ivy.stack([self._data] + arrays, axis, out=out)

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
        function, and so the docstring for ivy.clip also applies to this method
        with minimal changes.

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
        return ivy.clip(self._data, x_min=x_min, x_max=x_max, out=out)

    def constant_pad(
        self: ivy.Array,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        value: Optional[Number] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.constant_pad. This method simply
        wraps the function, and so the docstring for ivy.split also applies
        to this method with minimal changes.
        """
        return ivy.constant_pad(self._data, pad_width=pad_width, value=value, out=out)

    def repeat(
        self: ivy.Array,
        /,
        repeats: Union[int, Iterable[int]],
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.repeat. This method simply wraps the
        function, and so the docstring for ivy.repeat also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2.])
        >>> y= x.repeat(2)
        >>> print(y)
        ivy.array([0., 0., 1., 1., 2., 2.])
        """
        return ivy.repeat(self._data, repeats=repeats, axis=axis, out=out)

    def split(
        self: ivy.Array,
        /,
        *,
        num_or_size_splits: Optional[Union[int, Iterable[int]]] = None,
        axis: int = 0,
        with_remainder: bool = False,
    ) -> List[ivy.Array]:
        """
        ivy.Array instance method variant of ivy.split. This method simply
        wraps the function, and so the docstring for ivy.split also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            array to be divided into sub-arrays.
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers. Default
            is to divide into as many 1-dimensional arrays as the axis dimension.
        axis
            The axis along which to split, default is 0.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is False.

        Returns
        -------
            A list of sub-arrays.

        """
        return ivy.split(self._data, num_or_size_splits, axis, with_remainder)

    def swapaxes(
        self: ivy.Array,
        axis0: int,
        axis1: int,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.swap_axes. This method simply
        wraps the function, and so the docstring for ivy.split also applies
        to this method with minimal changes.
        """
        return ivy.swapaxes(self._data, axis0=axis0, axis1=axis1, out=out)

    def tile(
        self: ivy.Array,
        /,
        reps: Iterable[int],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tile. This method simply
        wraps the function, and so the docstring for ivy.tile also applies
        to this method with minimal changes.
        """
        return ivy.tile(self._data, reps=reps, out=out)

    def zero_pad(
        self: ivy.Array,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.zero_pad. This method simply
        wraps the function, and so the docstring for ivy.zero_pad also applies
        to this method with minimal changes.
        """
        return ivy.zero_pad(self._data, pad_width=pad_width, out=out)
