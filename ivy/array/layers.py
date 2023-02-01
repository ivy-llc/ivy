# global
import abc
from typing import Optional, Tuple, Union, List, Callable, Sequence

# local
import ivy


# ToDo: implement all methods here as public instance methods

# ToDo: update docstrings and typehints according to ivy\layers


class ArrayWithLayers(abc.ABC):
    def linear(
        self: ivy.Array,
        weight: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.linear. This method simply
        wraps the function, and so the docstring for ivy.linear also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input array to compute linear transformation on.
            *[outer_batch_shape,inner_batch_shape,in_features]*
        weight
            The weight matrix. *[outer_batch_shape,out_features,in_features]*
        bias
            The bias vector, default is ``None``. *[outer_batch_shape,out_features]*
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the linear transformation.
            *[outer_batch_shape,inner_batch_shape,out_features]*

        Examples
        --------
        >>> x = ivy.array([[1.1, 2.2, 3.3], \
                           [4.4, 5.5, 6.6], \
                           [7.7, 8.8, 9.9]])
        >>> w = ivy.array([[1., 2., 3.], \
                           [4., 5., 6.], \
                           [7., 8., 9.]])
        >>> b = ivy.array([1., 0., -1.])
        >>> y = x.linear(w, bias=b)
        >>> print(y)
        ivy.array([[ 16.4,  35.2,  54. ],
                   [ 36.2,  84.7, 133. ],
                   [ 56. , 134. , 212. ]])

        """
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
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        training: bool = True,
        seed: Optional[int] = None,
        noise_shape: Sequence[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.dropout. This method simply
        wraps the function, and so the docstring for ivy.droput also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input array x to perform dropout on.
        prob
            The probability of zeroing out each array element, float between 0 and 1.
        scale
            Whether to scale the output by `1/(1-prob)`, default is ``True``.
        dtype
            output array data type. If dtype is None, the output array data type
            must be inferred from x. Default: ``None``.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        seed
            Set a default seed for random number generating (for
            reproducibility).Default is ``None``.
        noise_shape
            a sequence representing the shape of the binary dropout mask that will be
            multiplied with the input.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the output after dropout is performed.

        Examples
        --------
        With :class:`ivy.Array` instances:

        >>> x = ivy.array([[1., 2., 3.],
        ...                [4., 5., 6.],
        ...                [7., 8., 9.],
        ...                [10., 11., 12.]])
        >>> y = x.dropout(0.3)
        >>> print(y)
        ivy.array([[ 1.42857146,  2.85714293,  4.28571415],
                   [ 5.71428585,  7.14285755,  8.5714283 ],
                   [ 0.        , 11.4285717 , 12.8571434 ],
                   [14.2857151 ,  0.        ,  0.        ]])

        >>> x = ivy.array([[1., 2., 3.],
        ...                [4., 5., 6.],
        ...                [7., 8., 9.],
        ...                [10., 11., 12.]])
        >>> y = x.dropout(0.3, scale=False)
        >>> print(y)
        ivy.array([[ 1.,  2., 3.],
                   [ 4.,  5., 0.],
                   [ 7.,  0., 9.],
                   [10., 11., 0.]])
        """
        return ivy.dropout(
            self._data,
            prob,
            scale=scale,
            dtype=dtype,
            training=training,
            seed=seed,
            noise_shape=noise_shape,
            out=out,
        )

    def dropout1d(
        self: ivy.Array,
        prob: float,
        /,
        *,
        training: bool = True,
        data_format: str = "NWC",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.dropout1d. This method simply
        wraps the function, and so the docstring for ivy.droput1d also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input array x to perform dropout on.
        prob
            The probability of zeroing out each array element, float between 0 and 1.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        data_format
            "NWC" or "NCW". Default is ``"NCW"``.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the output after dropout is performed.

        Examples
        --------
        >>> x = ivy.array([1, 1, 1]).reshape([1, 1, 3])
        >>> y = x.dropout1d(0.5)
        >>> print(y)
        ivy.array([[[2., 0, 2.]]])
        """
        return ivy.dropout1d(
            self._data,
            prob,
            training=training,
            data_format=data_format,
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
        """
        ivy.Array instance method variant of ivy.scaled_dot_product_attention.
        This method simply wraps the function, and so the docstring for
        ivy.scaled_dot_product_attention also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The queries input array. The shape of queries input array should be in
            *[batch_shape,num_queries,feat_dim]*. The queries input array should
            have the same size as keys and values.
        k
            The keys input array. The shape of keys input array should be in
            *[batch_shape,num_keys,feat_dim]*. The keys input array should have
            the same size as queries and values.
        v
            The values input array. The shape of values input should be in
            *[batch_shape,num_keys,feat_dim]*. The values input array should
            have the same size as queries and keys.
        scale
            The scale float value.
            The scale float value is used to scale the query-key pairs before softmax.
        mask
            The mask input array. The mask to apply to the query-key values.
            Default is None. The shape of mask input should be in
            *[batch_shape,num_queries,num_keys]*.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The output following application of scaled dot-product attention.
            The output array is the weighted sum produced by the attention score
            and value. The shape of output array is
            *[batch_shape,num_queries,feat_dim]* .

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> q = ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
        >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
        >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
        >>> mask = ivy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        >>> result = q.scaled_dot_product_attention(k, v, 1, mask=mask)
        >>> print(result)
        ivy.array([[[2.3, 3.23],[2.3, 3.23],[2.3, 3.23]]])

        >>> q = ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
        >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
        >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
        >>> mask = ivy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        >>> out = ivy.zeros(shape=(1, 3, 2))
        >>> q.scaled_dot_product_attention(k, v, 1, mask=mask, out=out)
        >>> print(out)
        ivy.array([[[2.3, 3.23],[2.3, 3.23],[2.3, 3.23]]])

        """
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
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        dilations: Union[int, Tuple[int]] = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.conv1d. This method simply
        wraps the function, and so the docstring for ivy.conv1d also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,w,d_in]* or *[batch_size,d_in,w]*.
        filters
            Convolution filters *[fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
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
        ivy.array([[[ 2.,  3.],
        ...         [ 4.,  7.],
        ...         [ 7., 13.],
        ...         [11., 20.]]])
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

    def conv2d(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.conv2d`. This method simply
        wraps the function, and so the docstring for `ivy.conv2d` also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d_in]*.
        filters
            Convolution filters *[fh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
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
        >>> x = ivy.array([[[[1.], [2.0],[3.]],
        ...                 [[1.], [2.0],[3.]],
        ...                 [[1.], [2.0],[3.]]]]) #NHWC
        >>> filters = ivy.array([[[[0.]], [[1.]], [[0.]]],
        ...                      [[[0.]], [[1.]], [[0.]]],
        ...                      [[[0.]], [[1.]], [[0.]]]]) #HWIO
        >>> result = x.conv2d(filters, 1, 'SAME', data_format='NHWC',
        ...    dilations= 1)
        >>> print(result)
        ivy.array([[
                  [[2.],[4.],[6.]],
                  [[3.],[6.],[9.]],
                  [[2.],[4.],[6.]]
                  ]])

        """
        return ivy.conv2d(
            self,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def conv2d_transpose(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int, int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        data_format: str = "NHWC",
        dilations: Union[int, Tuple[int, int]] = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.conv2d_transpose`. This method simply
        wraps the function, and so the docstring for `ivy.conv2d_transpose` also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d_in]* or *[batch_size,d_in,h,w]*.
        filters
            Convolution filters *[fh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        output_shape
            Shape of the output (Default value = None)
        data_format
            The ordering of the dimensions in the input, one of "NHWC" or "NCHW". "NHWC"
            corresponds to inputs with shape (batch_size, height, width, channels),
            while "NCHW" corresponds to input with shape (batch_size, channels, height,
            width). Default is ``"NHWC"``.
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the transpose convolution operation.

        Examples
        --------
        >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 28, 28, 3])
        >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 6])
        >>> y = x.conv2d_transpose(filters, 2, 'SAME')
        >>> print(y.shape)
        (1, 56, 56, 6)
        """
        return ivy.conv2d_transpose(
            self._data,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def conv3d(
        self: ivy.Array,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.conv3d`. This method simply
        wraps the function, and so the docstring for `ivy.conv3d` also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        filters
            Convolution filters *[fd,fh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NDHWC" or "NCDHW". Defaults to "NDHWC".
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
        >>> x = ivy.ones((1, 3, 3, 3, 1)).astype(ivy.float32)

        >>> filters = ivy.ones((1, 3, 3, 1, 1)).astype(ivy.float32)

        >>> result = x.conv3d(filters, 2, 'SAME')
        >>> print(result)
        ivy.array([[[[[4.],[4.]],[[4.],[4.]]],[[[4.],[4.]],[[4.],[4.]]]]])

        """
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
        """
        ivy.Array instance method variant of `ivy.conv3d_transpose`. This
        method simply wraps the function, and so the docstring for
        `ivy.conv3d_transpose` also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input volume *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
        filters
            Convolution filters *[fd,fh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        output_shape
            Shape of the output (Default value = None)
        data_format
            The ordering of the dimensions in the input, one of "NDHWC" or
            "NCDHW". "NDHWC" corresponds to inputs with shape (batch_size,
             depth, height, width, channels), while "NCDHW" corresponds
             to input with shape (batch_size, channels, depth, height,
             width).
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the transpose convolution operation.

        Examples
        --------
        >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
        >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])
        >>> y = x.conv3d_transpose(filters, 2, 'SAME')
        >>> print(y.shape)
        (1, 6, 56, 56, 6)
        """
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
        """
        ivy.Array instance method variant of ivy.lstm_update. This method simply
        wraps the function, and so the docstring for ivy.lstm_update also applies
        to this method with minimal changes.

        Parameters
        ----------
        init_h
            initial state tensor for the cell output *[batch_shape, out]*.
        init_c
            initial state tensor for the cell hidden state *[batch_shape, out]*.
        kernel
            weights for cell kernel *[in, 4 x out]*.
        recurrent_kernel
            weights for cell recurrent kernel *[out, 4 x out]*.
        bias
            bias for cell kernel *[4 x out]*. (Default value = None)
        recurrent_bias
            bias for cell recurrent kernel *[4 x out]*. (Default value = None)

        Returns
        -------
        ret
            hidden state for all timesteps *[batch_shape,t,out]* and cell state for last
            timestep *[batch_shape,out]*

        Examples
        --------
        >>> x = ivy.randint(0, 20, shape=(6, 20, 3))
        >>> h_i = ivy.random_normal(shape=(6, 5))
        >>> c_i = ivy.random_normal(shape=(6, 5))
        >>> kernel = ivy.random_normal(shape=(3, 4 * 5))
        >>> rc = ivy.random_normal(shape=(5, 4 * 5))
        >>> result = x.lstm_update(h_i, c_i, kernel, rc)

        >>> result[0].shape
        (6, 20, 5)
        >>> result[1].shape
        (6, 5)

        """
        return ivy.lstm_update(
            self._data,
            init_h,
            init_c,
            kernel,
            recurrent_kernel,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
