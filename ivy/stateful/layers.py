"""Collection of Ivy neural network layers as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module
from ivy.stateful.initializers import Zeros, GlorotUniform


# Linear #
# -------#


class Linear(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Linear layer, also referred to as dense or fully connected. The layer
        receives tensors with input_channels last dimension and returns a new tensor
        with output_channels last dimension, following matrix multiplication with the
        weight matrix and addition with the bias vector.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        with_bias
            Whether or not to include a bias term, default is True.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for the linear layer, as a container, constructed internally
            by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.


        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._w_shape = (output_channels, input_channels)
        self._b_shape = (output_channels,)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.



        """
        v = {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            )
        }
        if self._with_bias:
            v = dict(
                **v,
                b=self._b_init.create_variables(
                    self._b_shape, device, self._output_channels, dtype=dtype
                )
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the Linear layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_shape, in]*.

        Returns
        -------
        ret
            The outputs following the linear operation and bias addition
            *[batch_shape, out]*
        """
        return ivy.linear(inputs, self.v.w, bias=self.v.b if self._with_bias else None)


# Dropout #
# --------#


class Dropout(Module):
    def __init__(self, prob, scale=True, dtype=None):
        """
        Dropout layer. The layer randomly zeroes some of the elements of the input
        tensor with probability p using samples from a Bernoull distribution.

        Parameters
        ----------
        prob
            The probability of zeroing out each array element.
        scale
            Whether to scale the output by 1/(1-prob), default is True.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created.
            Default is None.
        """
        self._prob = prob
        self._scale = scale
        Module.__init__(self, None, None, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created .
            Default is None.


        """
        return {}

    def _forward(self, inputs, dtype=None):
        """
        Perform forward pass of the Linear layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_shape, in]*.
        dtype
            the desired data type of the internal variables to be created .
            Default is None.

        Returns
        -------
        ret
            The outputs following the linear operation and bias addition
            *[batch_shape, out]*
        """
        return ivy.dropout(inputs, self._prob, scale=self._scale, dtype=dtype)


# Attention #
# ----------#


class MultiHeadAttention(Module):
    def __init__(
        self,
        query_dim,
        num_heads=8,
        head_dim=64,
        dropout_rate=0.0,
        context_dim=None,
        scale=None,
        with_to_q_fn=True,
        with_to_kv_fn=True,
        with_to_out_fn=True,
        device=None,
        v=None,
        build_mode="on_init",
        dtype=None,
    ):
        """
        Multi Head Attention layer.

        Parameters
        ----------
        query_dim
            The dimension of the attention queries.
        num_heads
            Number of attention heads. Default is 8.
        head_dim
            The dimension of each of the heads. Default is 64.
        dropout_rate
            The rate of dropout. Default is 0.
        context_dim
            The dimension of the context array.
            Default is None, in which case the query dim is used.
        scale
            The value by which to scale the query-key similarity measure.
            Default is head_dim^-0.5
        with_to_q_fn
            Whether to include fully connected mapping from input x to queries.
            Default is True.
        with_to_kv_fn
            Whether to include fully connected mapping from input context to keys
            and values.
            Default is True.
        with_to_out_fn
            Whether to include fully connected mapping from output scaled dot-product
            attention to final output.
            Default is True.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for the attention layer, as a container,
            constructed internally by default.
        build_mode
            How the Module is built, either on initialization (now),
            explicitly by the user by calling
            build(), or the first time the __call__ method is run.
            Default is on initialization.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        v_exists = ivy.exists(v)
        v = ivy.default(v, ivy.Container({"to_q": None, "to_kv": None, "to_out": None}))
        self._query_dim = query_dim
        self._inner_dim = head_dim * num_heads
        self._dropout_rate = dropout_rate
        self._context_dim = ivy.default(context_dim, query_dim)
        self._scale = ivy.default(scale, head_dim**-0.5)
        self._num_heads = num_heads
        self._with_to_q_fn = with_to_q_fn
        self._with_to_kv_fn = with_to_kv_fn
        self._with_to_out_fn = with_to_out_fn
        ivy.Module.__init__(
            self,
            device,
            v if v_exists else None,
            build_mode,
            with_partial_v=True,
            dtype=dtype,
        )

    # noinspection PyAttributeOutsideInit
    def _build(self, *agrs, **kwargs):
        self._to_q = (
            ivy.Linear(
                self._query_dim, self._inner_dim, device=self._dev, dtype=self._dtype
            )
            if self._with_to_q_fn
            else None
        )
        self._to_k = (
            ivy.Linear(
                self._context_dim, self._inner_dim, device=self._dev, dtype=self._dtype
            )
            if self._with_to_kv_fn
            else None
        )
        self._to_v = (
            ivy.Linear(
                self._context_dim, self._inner_dim, device=self._dev, dtype=self._dtype
            )
            if self._with_to_kv_fn
            else None
        )
        self._to_kv = lambda context, v=None: (
            self._to_k(context, v=v.k if v else None),
            self._to_v(context, v=v.v if v else None),
        )
        self._to_out = (
            ivy.Sequential(
                ivy.Linear(
                    self._inner_dim,
                    self._query_dim,
                    device=self._dev,
                    dtype=self._dtype,
                ),
                ivy.Dropout(self._dropout_rate),
                device=self._dev,
            )
            if self._with_to_out_fn
            else None
        )

    def _create_variables(self, device, dtype=None):
        """
        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        return ivy.Container(to_kv={"k": self._to_k.v, "v": self._to_v.v})

    def _forward(self, inputs, context=None, mask=None):
        """
        Perform forward pass of the MultiHeadAttention layer.

        Parameters
        ----------
        inputs
            The array to determine the queries from *[batch_shape,num_queries,x_feats]*.
        context
            The array to determine the keys and values from. Default is None.
            *[batch_shape,num_values,cont_feats]*.
        mask
            (Default value = None)

        Returns
        -------
        ret
            The output following application of scaled dot-product attention.
            *[batch_shape,num_queries,out_feats]*
            The mask to apply to the query-key values.
            Default is None.
            *[batch_shape,num_queries,num_values]*
        """
        return ivy.multi_head_attention(
            inputs,
            self._scale,
            self._num_heads,
            context=context,
            mask=mask,
            to_q_fn=self._to_q,
            to_kv_fn=self._to_kv,
            to_out_fn=self._to_out,
            to_q_v=self.v.to_q,
            to_kv_v=self.v.to_kv,
            to_out_v=self.v.to_out,
        )


# Convolutions #
# -------------#


class Conv1D(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_size,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        data_format="NWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        1D convolutional layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        filter_size
            Size of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        data_format
            NWC" or "NCW". Defaults to "NWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_size = filter_size
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            (filter_size, input_channels, output_channels)
            if data_format == "NWC"
            else (input_channels, output_channels, self._filter_size)
        )
        self._b_shape = (1, 1, output_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created.
             Default is None.

        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._output_channels, dtype=dtype
            ),
        }

    def _forward(self, inputs):
        """
        Perform forward pass of the Conv1D layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,w,d_in]*

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_w,d_out]*

        """
        return (
            ivy.conv1d(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


class Conv1DTranspose(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_size,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        output_shape=None,
        data_format="NWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        1D transpose convolutional layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        filter_size
            Size of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        output_shape
            Shape of the output (Default value = None)
        data_format
            NWC" or "NCW". Defaults to "NWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_size = filter_size
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            (filter_size, input_channels, output_channels)
            if data_format == "NWC"
            else (input_channels, output_channels, filter_size)
        )
        self._b_shape = (1, 1, output_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._output_shape = output_shape
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._output_channels
            ),
        }

    def _forward(self, inputs):
        """Perform forward pass of the Conv1DTranspose layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,w,d_in]*

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_w,d_out]*

        """
        return (
            ivy.conv1d_transpose(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                output_shape=self._output_shape,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


class Conv2D(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        data_format="NHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """2D convolutional layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        filter_shape
            Shape of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        data_format
            NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            filter_shape + [input_channels, output_channels]
            if data_format == "NHWC"
            else [input_channels, output_channels] + filter_shape
        )
        self._b_shape = (1, 1, 1, output_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created.
            Default is None.

        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._output_channels, dtype=dtype
            ),
        }

    def _forward(self, inputs):
        """Perform forward pass of the Conv2D layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_h,new_w,d_out]*

        """
        return (
            ivy.conv2d(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


class Conv2DTranspose(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        output_shape=None,
        data_format="NHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """2D convolutional transpose layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        filter_shape
            Shape of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        output_shape
            Shape of the output (Default value = None)
        data_format
            NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            filter_shape + [input_channels, output_channels]
            if data_format == "NHWC"
            else [input_channels, output_channels] + filter_shape
        )
        self._b_shape = (1, 1, 1, output_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._output_shape = output_shape
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._output_channels, dtype=dtype
            ),
        }

    def _forward(self, inputs):
        """Perform forward pass of the Conv2DTranspose layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_h,new_w,d_out]*

        """
        return (
            ivy.conv2d_transpose(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                output_shape=self._output_shape,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


class DepthwiseConv2D(Module):
    def __init__(
        self,
        num_channels,
        filter_shape,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        data_format="NHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Depthwise 2D convolutional layer.

        Parameters
        ----------
        num_channels
            Number of input channels for the layer.
        filter_shape
            Shape of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        data_format
            NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._num_channels = num_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            filter_shape + [num_channels]
            if data_format == "NHWC"
            else [num_channels] + filter_shape
        )
        self._b_shape = (1, 1, num_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._num_channels,
                self._num_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._num_channels, dtype=dtype
            ),
        }

    def _forward(self, inputs):
        """Perform forward pass of the DepthwiseConv2D layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_h,new_w,d_out]*

        """
        return (
            ivy.depthwise_conv2d(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


class Conv3D(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        data_format="NDHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """3D convolutional layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        filter_shape
            Shape of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            filter_shape + [input_channels, output_channels]
            if data_format == "NDHWC"
            else [input_channels, output_channels] + filter_shape
        )
        self._b_shape = (1, 1, 1, 1, output_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._output_channels, dtype=dtype
            ),
        }

    def _forward(self, inputs):
        """Perform forward pass of the Conv3D layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,d,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer
            *[batch_size,new_d,new_h,new_w,d_out]*
        """
        return (
            ivy.conv3d(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


class Conv3DTranspose(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        output_shape=None,
        data_format="NDHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """3D convolutional transpose layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        filter_shape
            Shape of the convolutional filter.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        bias_initializer
            Initializer for the bias. Default is Zeros.
        output_shape
            Shape of the output (Default value = None)
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the linear layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = (
            filter_shape + [input_channels, output_channels]
            if data_format == "NDHWC"
            else [input_channels, output_channels] + filter_shape
        )
        self._b_shape = (1, 1, 1, 1, output_channels)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._output_shape = output_shape
        self._data_format = data_format
        self._dilations = dilations
        self.dtype = dtype
        Module.__init__(self, device, v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        return {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._output_channels,
                self._input_channels,
                dtype=dtype,
            ),
            "b": self._b_init.create_variables(
                self._b_shape, device, self._output_channels, dtype=dtype
            ),
        }

    def _forward(self, inputs):
        """Perform forward pass of the Conv3DTranspose layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,d,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer
            *[batch_size,new_d,new_h,new_w,d_out]*
        """
        return (
            ivy.conv3d_transpose(
                inputs,
                self.v.w,
                self._strides,
                self._padding,
                output_shape=self._output_shape,
                data_format=self._data_format,
                dilations=self._dilations,
            )
            + self.v.b
        )


# LSTM #
# -----#


class LSTM(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        weight_initializer=GlorotUniform(),
        num_layers=1,
        return_sequence=True,
        return_state=True,
        device=None,
        v=None,
        dtype=None,
    ):
        """LSTM layer, which is a set of stacked lstm cells.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer
        output_channels
            Number of output channels for the layer
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        num_layers
            Number of lstm cells in the lstm layer, default is 1.
        return_sequence
            Whether or not to return the entire output sequence, or
            just the latest timestep.
            Default is True.
        return_state
            Whether or not to return the latest hidden and cell states. Default is True.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the lstm cells, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._w_init = weight_initializer
        self._num_layers = num_layers
        self._return_sequence = return_sequence
        self._return_state = return_state
        Module.__init__(self, device, v, dtype=dtype)

    # Public #

    def get_initial_state(self, batch_shape, dtype=None):
        """Get the initial state of the hidden and cell states, if not provided
        explicitly

        Parameters
        ----------
        batch_shape
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        batch_shape = list(batch_shape)
        return (
            [
                ivy.zeros((batch_shape + [self._output_channels]), dtype=dtype)
                for i in range(self._num_layers)
            ],
            [
                ivy.zeros((batch_shape + [self._output_channels]), dtype=dtype)
                for i in range(self._num_layers)
            ],
        )

    # Overridden

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is None.

        """
        input_weights = dict(
            zip(
                ["layer_" + str(i) for i in range(self._num_layers)],
                [
                    {
                        "w": self._w_init.create_variables(
                            (
                                self._input_channels
                                if i == 0
                                else self._output_channels,
                                4 * self._output_channels,
                            ),
                            device,
                            self._output_channels,
                            self._input_channels,
                            dtype=dtype,
                        )
                    }
                    for i in range(self._num_layers)
                ],
            )
        )
        recurrent_weights = dict(
            zip(
                ["layer_" + str(i) for i in range(self._num_layers)],
                [
                    {
                        "w": self._w_init.create_variables(
                            (self._output_channels, 4 * self._output_channels),
                            device,
                            self._output_channels,
                            self._input_channels,
                            dtype=dtype,
                        )
                    }
                    for i in range(self._num_layers)
                ],
            )
        )
        return {"input": input_weights, "recurrent": recurrent_weights}

    def _forward(self, inputs, initial_state=None):
        """Perform forward pass of the LSTM layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_shape, t, in]*.
        initial_state
            2-tuple of lists of the hidden states h and c for each layer,
            each of dimension *[batch_shape,out]*.
            Created internally if None. (Default value = None)

        Returns
        -------
        ret
            The outputs of the final lstm layer *[batch_shape, t, out]* and the hidden
            state tuple of lists, each of dimension *[batch_shape, out]*

        """
        if initial_state is None:
            initial_state = self.get_initial_state(
                inputs.shape[:-2], dtype=inputs.dtype
            )
        h_n_list = list()
        c_n_list = list()
        h_t = inputs
        for h_0, c_0, (_, lstm_input_var), (_, lstm_recurrent_var) in zip(
            initial_state[0],
            initial_state[1],
            self.v.input.items(),
            self.v.recurrent.items(),
        ):
            h_t, c_n = ivy.lstm_update(
                h_t, h_0, c_0, lstm_input_var.w, lstm_recurrent_var.w
            )
            h_n_list.append(h_t[..., -1, :])
            c_n_list.append(c_n)
        if not self._return_sequence:
            h_t = h_t[..., -1, :]
        if not self._return_state:
            return h_t
        return h_t, (h_n_list, c_n_list)
