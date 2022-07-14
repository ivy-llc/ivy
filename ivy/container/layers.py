# global
from typing import Optional, Tuple, Union, List

# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithLayers(ContainerBase):
    @staticmethod
    def static_depthwise_conv2d(
        x: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.depthwise_conv2d. This method
        simply wraps the function, and so the docstring for ivy.depthwise_conv2d
        also applies to this method with minimal changes.

        Examples
        --------
        >>> a = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> b = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> inp = ivy.Container(a=a, b=b)
        >>> filters = ivy.random_normal(0, 1, [3, 3, 3])
        >>> y = ivy.Container.static_depthwise_conv2d( \
                                                    inp, \
                                                    filters, \
                                                    strides=2, \
                                                    padding='SAME')
        >>> print(y.shape)
        [1, 64, 64, 3]
        """
        return ContainerBase.multi_map_in_static_method(
            "depthwise_conv2d",
            x,
            filters=filters,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def depthwise_conv2d(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.depthwise_conv2d. This method
        simply wraps the function, and so the docstring for ivy.depthwise_conv2d
        also applies to this method with minimal changes.

        Examples
        --------
        >>> a = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> b = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> inp = ivy.Container(a=a, b=b)
        >>> filters = ivy.random_normal(0, 1, [3, 3, 3])
        >>> y = inp.depthwise_conv2d(filters, strides=2, padding='SAME')
        >>> print(y.shape)
        [1, 64, 64, 3]
        """
        return self.static_depthwise_conv2d(
            self, filters, strides, padding, data_format, dilations, out=out
        )
