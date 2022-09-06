# global
import abc
from typing import Optional, Tuple, Union, List, Callable

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithLayers(abc.ABC):
    def linear(
        self: ivy.Array,
        weight: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        bias: Union[ivy.Array, ivy.NativeArray] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.linear(
            self._data,
            weight,
            bias=bias,
            out=out,
        )

    def dropout(
        self: ivy.Array,
        prob: float,
        /,
        *,
        scale: bool = True,
        dtype: ivy.Dtype = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.dropout(
            self._data,
            prob,
            scale=scale,
            dtype=dtype,
            out=out,
        )

    def scaled_dot_product_attention(
        self: ivy.Array,
        k: Union[ivy.Array, ivy.NativeArray],
        v: Union[ivy.Array, ivy.NativeArray],
        scale: float,
        /,
        *,
        mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.scaled_dot_product_attention(
            self._data,
            k,
            v,
            scale,
            mask=mask,
            out=out,
        )

    def multi_head_attention(
        self: ivy.Array,
        scale,
        num_heads,
        /,
        *,
        context: Union[ivy.Array, ivy.NativeArray] = None,
        mask: Union[ivy.Array, ivy.NativeArray] = None,
        to_q_fn: Callable = None,
        to_kv_fn: Callable = None,
        to_out_fn: Callable = None,
        to_q_v=None,
        to_kv_v=None,
        to_out_v=None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.multi_head_attention(
            self._data,
            scale,
            num_heads,
            context=context,
            mask=mask,
            to_q_fn=to_q_fn,
            to_kv_fn=to_kv_fn,
            to_out_fn=to_out_fn,
            to_q_v=to_q_v,
            to_kv_v=to_kv_v,
            to_out_v=to_out_v,
            out=out,
        )

    def conv1d(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: int,
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        dilations: int = 1,
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
            SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            NWC" or "NCW". Defaults to "NWC".
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
        >>> x = ivy.array([[[1., 2.], [3., 4.], [6., 7.], [9., 11.]]])  # NWC
        >>> filters = ivy.array([[[0., 1.], [1., 1.]]])  # WIO (I == C)
        >>> result = x.conv1d(filters, (1,), 'VALID')
        >>> print(result)
        ivy.array([[[ 2.,  3.], \
                    [ 4.,  7.], \
                    [ 7., 13.], \
                    [11., 20.]]])
        """
        return ivy.conv1d(
            self._data,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def conv1d_transpose(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: int,
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        data_format: str = "NWC",
        dilations: int = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.conv1d_transpose(
            self._data,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def depthwise_conv2d(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
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
        >>> x = ivy.randint(0, 255, shape=(1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3])
        >>> y = x.depthwise_conv2d(filters, 2, 'SAME')
        >>> print(y.shape)
        (1, 64, 64, 3)
        """
        return ivy.depthwise_conv2d(
            self._data,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def conv3d(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: int,
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        dilations: int = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.conv3d(
            self._data,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def conv3d_transpose(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        data_format: str = "NDHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.conv3d_transpose(
            self._data,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def lstm_update(
        self: ivy.Array,
        init_h: Union[ivy.Array, ivy.NativeArray],
        init_c: Union[ivy.Array, ivy.NativeArray],
        kernel: Union[ivy.Array, ivy.NativeArray],
        recurrent_kernel: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        recurrent_bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Tuple[ivy.Array, ivy.Array]:
        return ivy.lstm_update(
            self._data,
            init_h,
            init_c,
            kernel,
            recurrent_kernel,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
