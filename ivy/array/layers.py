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


    def lstm_update(
        self: ivy.Array,
        hidden_state: Union[ivy.Array, ivy.NativeArray],
        cell_state: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lstm_update. This method simply wraps
        the function, and so the docstring for ivy.lstm_update also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d]*.
        hidden_state
            Hidden state *[batch_size,h,w,d]*.
        cell_state
            Cell state *[batch_size,h,w,d]*.
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
        >>> hidden_state = ivy.random_normal(0, 1, [1, 128, 128, 64])
        >>> cell_state = ivy.random_normal(0, 1, [1, 128, 128, 64])
        >>> y = x.lstm_update(hidden_state, cell_state)
        >>> print(y.shape)
        (1, 128, 128, 64)
        """
        return ivy.lstm_update(self._data, hidden_state, cell_state, out=out)
    