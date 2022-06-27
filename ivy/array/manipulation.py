# global
import abc
from typing import Optional, Union, Tuple, List, Iterable
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
        axis: Optional[int] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.concat([self._data] + xs, axis, out=out)

    def flip(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.flip(self._data, axis, out=out)

    def expand_dims(
        self: ivy.Array,
        axis: Optional[int] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.expand_dims(self._data, axis, out=out)

    def reshape(
        self: ivy.Array,
        shape: Tuple[int, ...],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.reshape(self._data, shape, out=out)

    def permute_dims(
        self: ivy.Array,
        axes: Tuple[int, ...],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.permute_dims(self._data, axes, out=out)

    def roll(
        self: ivy.Array,
        shift: Union[int, Tuple[int, ...]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.roll. This method simply wraps the
        function, and so the docstring for ivy.roll also applies to this method
        with minimal changes.

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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.squeeze(self._data, axis=axis, out=out)

    def stack(
        self: ivy.Array,
        x: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray]],
            List[Union[ivy.Array, ivy.NativeArray]],
        ],
        axis: Optional[int] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.stack([self._data] + x, axis, out=out)

    def clip(
        x: Union[ivy.Array, ivy.NativeArray],
        x_min: Union[Number, Union[ivy.Array, ivy.NativeArray]],
        x_max: Union[Number, Union[ivy.Array, ivy.NativeArray]],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Examples
        --------
        With :code:`ivy.Array` input:

        >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> y = ivy.clip(x, 1., 5.)
        >>> print(y)
        ivy.array([1., 1., 2., 3., 4., 5., 5., 5., 5., 5.])

        >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> y = ivy.zeros_like(x)
        >>> ivy.clip(x, 2., 7., out=y)
        >>> print(y)
        ivy.array([2., 2., 2., 3., 4., 5., 6., 7., 7., 7.])

        >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> x_min = ivy.array([3., 4., 1., 0., 2., 3., 4., 4., 4., 4.])
        >>> x_max = ivy.array([5., 4., 3., 3., 5., 7., 8., 3., 8., 8.])
        >>> y = ivy.clip(x, x_min, x_max)
        >>> print(y)
        ivy.array([3., 4., 2., 3., 4., 5., 6., 3., 8., 8.])

        With :code:`ivy.NativeArray` input:

        >>> x = ivy.native_array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> x_min = ivy.native_array([3., 4., 1., 0., 2., 3., 4., 4., 4., 4.])
        >>> x_max = ivy.native_array([5., 4., 3., 3., 5., 7., 8., 3., 8., 8.])
        >>> y = ivy.clip(x, x_min, x_max)
        >>> print(y)
        ivy.array([3., 4., 2., 3., 4., 5., 6., 3., 8., 8.])

        Instance Method Examples
        ------------------------

        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> y = x.clip(1., 5.)
        >>> print(y)
        ivy.array([1., 1., 2., 3., 4., 5., 5., 5., 5., 5.])
        """
        return current_backend(x).squeeze(x, axis, out=out)


    def repeats(
        self: ivy.Array,
        repeats: Union[int, Iterable[int]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.repeat(self._data, repeats=repeats, axis=axis, out=out)

    def tile(
        self: ivy.Array,
        reps: Iterable[int],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.tile(self._data, reps=reps, out=out)

    def constant_pad(
        self: ivy.Array,
        pad_width: Iterable[Tuple[int]],
        value: Number = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.constant_pad(self._data, pad_width=pad_width, value=value, out=out)

    def zero_pad(
        self: ivy.Array,
        pad_width: Iterable[Tuple[int]],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.zero_pad(self._data, pad_width=pad_width, out=out)

    def swapaxes(
        self: ivy.Array,
        axis0: int,
        axis1: int,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.swapaxes(self._data, axis0=axis0, axis1=axis1, out=out)
