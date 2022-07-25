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
        ivy.Array instance method variant of ivy.depthwise_conv2d. This method simply
        wraps the function, and so the docstring for ivy.depthwise_conv2d also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d]*.
        filters
            Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from self)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

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

    def conv1d(
        self: ivy.Array,
            filters: Union[ivy.Array, ivy.NativeArray],
            strides: int,
            padding: str,
            data_format: str = "NWC",
            dilations: int = 1,
            *,
            out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.conv1d. This method simply
        wraps the function, and so the docstring for ivy.conv1d also applies
        to this method with minimal changes.

       Parameters
    ----------
    x
        Input image *[batch_size,w,d_in]*.
    filters
        Convolution filters *[fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        NWC" or "NCW". Defaults to "NWC".
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.array(ivy.random_normal(0, 1, [2, 12, 2]))  # NWC
        >>> filters = ivy.array(ivy.random_normal(0, 1, [6,2, 2]))  # WIO (I == C)
        >>> result = x.conv1d(filters, strides=(3,), padding='VALID')
        >>> print(result)

        ivy.array([[[ 0.365,  1.82 ],
                    [-1.79 ,  1.7  ],
                    [ 2.91 , -4.01 ]],

                   [[-5.08 , -2.4  ],
                    [ 0.387, -1.61 ],
                    [-2.14 , -1.46 ]]])

        """
        return ivy.conv1d(
            self._data, filters, strides, padding, data_format, dilations, out=out
        )
