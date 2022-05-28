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
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        axis: Optional[int] = 0,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.concat([self] + xs, axis, out=out)

    def flip(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Array:
        return ivy.flip(self, axis, out=out)

    def expand_dims(
        self: ivy.Array,
        axis: Optional[int] = 0,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.expand_dims(self, axis, out=out)

    def reshape(
        self: ivy.Array,
        shape: Tuple[int, ...],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.reshape(self, shape, out=out)

    def permute_dims(
        self: ivy.Array,
        axes: Tuple[int, ...],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.permute_dims(self, axes, out=out)

    def roll(
        self: ivy.Array,
        shift: Union[int, Tuple[int, ...]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.roll(self, shift=shift, axis=axis, out=out)

    def squeeze(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.squeeze(self, axis=axis, out=out)

    def stack(
        self: ivy.Array,
        x: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        axis: Optional[int] = 0,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.stack([self] + x, axis, out=out)

    def repeats(
        self: ivy.Array,
        repeats: Union[int, Iterable[int]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.repeat(self, repeats=repeats, axis=axis, out=out)

    def tile(
        self: ivy.Array,
        reps: Iterable[int],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.tile(self, reps=reps, out=out)

    def constant_pad(
        self: ivy.Array,
        pad_width: Iterable[Tuple[int]],
        value: Number = 0,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.constant_pad(self, pad_width=pad_width, value=value, out=out)

    def zero_pad(
        self: ivy.Array,
        pad_width: Iterable[Tuple[int]],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.zero_pad(self, pad_width=pad_width, out=out)

    def swapaxes(
        self: ivy.Array,
        axis0: int,
        axis1: int,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.swapaxes(self, axis0=axis0, axis1=axis1, out=out)
