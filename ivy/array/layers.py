# global
import abc
from typing import Optional, Tuple, Union, List

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithLayers(abc.ABC):
    def depthwise_conv2d(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.depthwise_conv2d. This method
        simply wraps the function, and so the docstring for ivy.depthwise_conv2d
        also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> filters = ivy.random_normal(0, 1, [3, 3, 3])
        >>> y = x.depthwise_conv2d(filters, strides=2, padding='SAME')
        >>> print(y.shape)
        (1, 64, 64, 3)
        """
        return ivy.depthwise_conv2d(
            self._data, filters, strides, padding, data_format, dilations, out=out
        )
