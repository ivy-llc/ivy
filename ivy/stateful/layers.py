"""Collection of Ivy neural network layers as stateful classes."""
# flake8: noqa
# local
import ivy
from ivy.func_wrapper import handle_nestable
from ivy.stateful.initializers import GlorotUniform, RandomNormal, Zeros
from ivy.stateful.module import Module

# ToDo: update docstrings and typehints according to ivy\layers


# Linear #
# -------#


class Linear(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Linear layer, also referred to as dense or fully connected. The layer receives
        tensors with input_channels last dimension and returns a new tensor with
        output_channels last dimension, following matrix multiplication with the weight
        matrix and addition with the bias vector.

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
            Whether or not to include a bias term, default is ``True``.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for the linear layer, as a container, constructed internally
            by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._w_shape = (output_channels, input_channels)
        self._b_shape = (output_channels,)
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, x):
        """
        Perform forward pass of the Linear layer.

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, in]*.

        Returns
        -------
        ret
            The outputs following the linear operation and bias addition
            *[batch_shape, out]*
        """
        return ivy.linear(x, self.v.w, bias=self.v.b if self._with_bias else None)


# Dropout #
# --------#


class Dropout(Module):
    def __init__(
        self,
        prob,
        scale: bool = True,
        dtype=None,
        training: bool = True,
    ):
        """
        Dropout layer. The layer randomly zeroes some of the elements of the input
        tensor with probability p using samples from a Bernoull distribution.

        Parameters
        ----------
        prob
            The probability of zeroing out each array element.
        scale
            Whether to scale the output by 1/(1-prob), default is ``True``.
        dtype
            the desired data type of the internal variables to be created.
            Default is ``None``.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        """
        self._prob = prob
        self._scale = scale
        self.training = training
        Module.__init__(self, device=None, v=None, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created .
            Default is ``None``.
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
            Default is ``None``.

        Returns
        -------
        ret
            The outputs following the linear operation and bias addition
            *[batch_shape, out]*
        """
        return ivy.dropout(
            inputs, self._prob, scale=self._scale, training=self.training, dtype=dtype
        )


# Attention #
# ----------#


class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim=None,
        /,
        *,
        key_dim=None,
        value_dim=None,
        num_heads=8,
        head_dim=None,
        dropout_rate=0.0,
        use_proj_bias=True,
        attention_axes=None,
        scale=None,
        device=None,
        v=None,
        build_mode="on_init",
        dtype=None,
    ):
        """
        Multi Head Attention layer.

        Parameters
        ----------
        embed_dim
            The expected feature size in the input and output.
        key_dim
            The input feature size for key. If None, assumed equal to `embed_dim`.
            Default None.
        value_dim
            The input feature size for value. If None, assumed equal to `embed_dim`.
            Default None.
        num_heads:
            Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
            Default is 8.
        head_dim
            Size of each attention head for query and key.
            Note that only two out of (``embed_dim``, ``num_heads``, and ``head_dim``) should be provided
            Default is None.
        dropout_rate
            The dropout probability used on attention weights to drop some attention targets. 0 for no dropout.
            Default is 0.
        use_proj_bias
            If specified, adds bias to input / output projection layers.
            Default is True.
        attention_axes
            axes over which the attention is applied. `None` means attention over all axes, but batch, heads, and features.
            Default is None.
        scale
            The value by which to scale the query-key similarity measure.
            Default is head_dim^-0.5
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
            Default is cpu.
        v
            the variables for the attention layer, as a container,
            constructed internally by default.
        build_mode
            How the Module is built, either on initialization (now),
            explicitly by the user by calling
            build(), or the first time the __call__ method is run.
            Default is on initialization.
        dtype
            the desired data type of the internal variables to be created if not provided.
            Default is ``None``.
        """
        # proj

        if num_heads and head_dim:
            self._inner_dim = num_heads * head_dim
        else:
            self._inner_dim = embed_dim

        self._embed_dim = embed_dim if embed_dim else num_heads * head_dim
        self._key_dim = key_dim if key_dim else self._embed_dim
        self._value_dim = value_dim if value_dim else self._embed_dim
        self._num_heads = num_heads if num_heads else embed_dim // head_dim
        self._head_dim = head_dim if head_dim else embed_dim // num_heads
        self._dropout_rate = dropout_rate
        self._use_proj_bias = use_proj_bias
        self._attention_axes = attention_axes
        self._scale = ivy.default(scale, self._head_dim**-0.5)
        self._qkv_same_embed_dim = (
            self._key_dim == self._embed_dim and self._value_dim == self._embed_dim
        )
        ivy.Module.__init__(
            self,
            device=device,
            v=v,
            build_mode=build_mode,
            with_partial_v=True,
            dtype=dtype,
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
             provided. Default is ``None``.
        """
        v = dict(
            out_proj_weights=GlorotUniform().create_variables(
                (self._embed_dim, self._inner_dim),
                device,
                self._embed_dim,
                self._inner_dim,
                dtype=dtype,
            ),
        )
        if self._qkv_same_embed_dim:
            v = dict(
                **v,
                in_proj_weights=GlorotUniform().create_variables(
                    (self._inner_dim * 3, self._embed_dim),
                    device,
                    self._inner_dim * 3,
                    self._embed_dim,
                    dtype=dtype,
                ),
            )
        else:
            v = dict(
                **v,
                q_proj_weights=GlorotUniform().create_variables(
                    (self._inner_dim, self._embed_dim),
                    device,
                    self._inner_dim,
                    self._embed_dim,
                    dtype=dtype,
                ),
                k_proj_weights=GlorotUniform().create_variables(
                    (self._inner_dim, self._key_dim),
                    device,
                    self._inner_dim,
                    self._key_dim,
                    dtype=dtype,
                ),
                v_proj_weights=GlorotUniform().create_variables(
                    (self._inner_dim, self._value_dim),
                    device,
                    self._inner_dim,
                    self._value_dim,
                    dtype=dtype,
                ),
            )
        if self._use_proj_bias:
            v = dict(
                **v,
                in_proj_bias=Zeros().create_variables(
                    self._inner_dim * 3,
                    device,
                    dtype=dtype,
                ),
                out_proj_bias=Zeros().create_variables(
                    self._embed_dim,
                    device,
                    dtype=dtype,
                ),
            )

        return v

    def _forward(
        self,
        query,
        key=None,
        value=None,
        /,
        *,
        attention_mask=None,
        is_causal=False,
        return_attention_weights=False,
        average_attention_weights=True,
        training=False,
    ):
        """
        Perform forward pass of the MultiHeadAttention layer.

        Parameters
        ----------
        query
            query embeddings *[batch_shape,num_queries,query_dim]*.
        key
            key embeddings *[batch_shape,num_queries,key_dim]*.
        value
            value embeddings *[batch_shape,num_queries,value_dim]*.
        attention_mask
            The mask to apply to the query-key values. Default is ``None``.
            *[batch_shape,num_queries,num_keys]*.
        is_causal
            If True, Uses a causal attention mask and ignores provided attention_mask.
        return_attention_weights
            If True, returns attention_weights alongside the output
            as a tuple (output, attenion_weights). Defaults to `False`.
        average_attention_weights
            If true, indicates that the returned ``attention_weights`` should be averaged across
            heads. Otherwise, ``attention_weights`` are provided separately per head. Note that this flag only has an
            effect when ``return_attention_weights=True``. Default: ``True`` (i.e. average weights across heads)
        training
            If True, dropout is used, otherwise dropout is not activated.

        Returns
        -------
        ret
            The output following application of multi-head attention.
            *[batch_shape,num_queries,out_feat_dim]* if input is batched
            otherwise *[num_queries, out_feat_dim]
        """
        return ivy.multi_head_attention(
            query,
            key,
            value,
            num_heads=self._num_heads,
            scale=self._scale,
            attention_mask=attention_mask,
            in_proj_weights=(
                self.v.in_proj_weights if self._qkv_same_embed_dim else None
            ),
            q_proj_weights=(
                self.v.q_proj_weights if not self._qkv_same_embed_dim else None
            ),
            k_proj_weights=(
                self.v.k_proj_weights if not self._qkv_same_embed_dim else None
            ),
            v_proj_weights=(
                self.v.v_proj_weights if not self._qkv_same_embed_dim else None
            ),
            out_proj_weights=self.v.out_proj_weights,
            in_proj_bias=self.v.in_proj_bias if self._use_proj_bias else None,
            out_proj_bias=self.v.out_proj_bias if self._use_proj_bias else None,
            is_causal=is_causal,
            return_attention_weights=return_attention_weights,
            average_attention_weights=average_attention_weights,
            dropout=self._dropout_rate,
            training=training,
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
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
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
        with_bias
            Whether or not to include a bias term, default is ``True``.
        data_format
            NWC" or "NCW". Defaults to "NWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_size = filter_size
        self._strides = strides
        self._padding = padding
        self._w_shape = (filter_size, input_channels, output_channels)
        self._b_shape = (
            (1, 1, output_channels) if data_format == "NWC" else (1, output_channels, 1)
        )
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created.
             Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

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
        return ivy.conv1d(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


class Conv1DTranspose(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_size,
        strides,
        padding,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
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
        with_bias
            Whether or not to include a bias term, default is ``True``.
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
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_size = filter_size
        self._strides = strides
        self._padding = padding
        self._w_shape = (filter_size, input_channels, output_channels)
        self._b_shape = (
            (1, 1, output_channels) if data_format == "NWC" else (1, output_channels, 1)
        )
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        self._output_shape = output_shape
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the Conv1DTranspose layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,w,d_in]*

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_w,d_out]*
        """
        return ivy.conv1d_transpose(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            output_shape=self._output_shape,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


class Conv2D(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
        data_format="NHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        2D convolutional layer.

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
        with_bias
            Whether or not to include a bias term, default is ``True``.
        data_format
            NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = filter_shape + [input_channels, output_channels]
        self._b_shape = (
            (1, 1, 1, output_channels)
            if data_format == "NHWC"
            else (1, output_channels, 1, 1)
        )
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created.
            Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the Conv2D layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_h,new_w,d_out]*
        """
        return ivy.conv2d(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


class Conv2DTranspose(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
        output_shape=None,
        data_format="NHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        2D convolutional transpose layer.

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
        with_bias
            Whether or not to include a bias term, default is ``True``.
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
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = filter_shape + [input_channels, output_channels]
        self._b_shape = (
            (1, 1, 1, output_channels)
            if data_format == "NHWC"
            else (1, output_channels, 1, 1)
        )
        self._w_init = weight_initializer
        self._with_bias = with_bias
        self._b_init = bias_initializer
        self._output_shape = output_shape
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the Conv2DTranspose layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_h,new_w,d_out]*
        """
        return ivy.conv2d_transpose(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            output_shape=self._output_shape,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


class DepthwiseConv2D(Module):
    def __init__(
        self,
        num_channels,
        filter_shape,
        strides,
        padding,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
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
        with_bias
            Whether or not to include a bias term, default is ``True``.
        data_format
            NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._num_channels = num_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = filter_shape + [num_channels]
        self._b_shape = (
            (1, 1, 1, num_channels)
            if data_format == "NHWC"
            else (1, num_channels, 1, 1)
        )
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        v = {
            "w": self._w_init.create_variables(
                self._w_shape,
                device,
                self._num_channels,
                self._num_channels,
                dtype=dtype,
            )
        }
        if self._with_bias:
            v = dict(
                **v,
                b=self._b_init.create_variables(
                    self._b_shape,
                    device,
                    self._num_channels,
                    self._num_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the DepthwiseConv2D layer.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_size,h,w,d_in]*.

        Returns
        -------
        ret
            The outputs following the conv1d layer *[batch_size,new_h,new_w,d_out]*
        """
        return ivy.depthwise_conv2d(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


class Conv3D(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
        data_format="NDHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        3D convolutional layer.

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
        with_bias
            Whether or not to include a bias term, default is ``True``.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = filter_shape + [input_channels, output_channels]
        self._b_shape = (
            (1, 1, 1, 1, output_channels)
            if data_format == "NDHWC"
            else (1, output_channels, 1, 1, 1)
        )
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        self._data_format = data_format
        self._dilations = dilations
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the Conv3D layer.

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
        return ivy.conv3d(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


class Conv3DTranspose(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_shape,
        strides,
        padding,
        /,
        *,
        weight_initializer=GlorotUniform(),
        bias_initializer=Zeros(),
        with_bias=True,
        output_shape=None,
        data_format="NDHWC",
        dilations=1,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        3D convolutional transpose layer.

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
        with_bias
            Whether or not to include a bias term, default is ``True``.
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
            the variables for each of the conv layer, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._w_shape = filter_shape + [input_channels, output_channels]
        self._b_shape = (
            (1, 1, 1, 1, output_channels)
            if data_format == "NDHWC"
            else (1, output_channels, 1, 1, 1)
        )
        self._w_init = weight_initializer
        self._b_init = bias_initializer
        self._with_bias = with_bias
        self._output_shape = output_shape
        self._data_format = data_format
        self._dilations = dilations
        self.dtype = dtype
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
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
                    self._b_shape,
                    device,
                    self._output_channels,
                    self._input_channels,
                    dtype=dtype,
                ),
            )
        return v

    def _forward(self, inputs):
        """
        Perform forward pass of the Conv3DTranspose layer.

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
        return ivy.conv3d_transpose(
            inputs,
            self.v.w,
            self._strides,
            self._padding,
            output_shape=self._output_shape,
            data_format=self._data_format,
            dilations=self._dilations,
        ) + (self.v.b if self._with_bias else 0)


# LSTM #
# -----#


class LSTM(Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        /,
        *,
        weight_initializer=GlorotUniform(),
        num_layers=1,
        return_sequence=True,
        return_state=True,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        LSTM layer, which is a set of stacked lstm cells.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer
        output_channels
            Number of output channels for the layer
        weight_initializer
            Initializer for the weights. Default is GlorotUniform.
        num_layers
            Number of lstm cells in the lstm layer, default is ``1``.
        return_sequence
            Whether or not to return the entire output sequence, or
            just the latest timestep.
            Default is ``True``.
        return_state
            Whether or not to return the latest hidden and cell states.
            Default is ``True``.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        v
            the variables for each of the lstm cells, as a container,
            constructed internally by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._w_init = weight_initializer
        self._num_layers = num_layers
        self._return_sequence = return_sequence
        self._return_state = return_state
        Module.__init__(self, device=device, v=v, dtype=dtype)

    # Public #

    def get_initial_state(self, batch_shape, dtype=None):
        """
        Get the initial state of the hidden and cell states, if not provided explicitly.

        Parameters
        ----------
        batch_shape
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
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
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        input_weights = dict(
            zip(
                ["layer_" + str(i) for i in range(self._num_layers)],
                [
                    {
                        "w": self._w_init.create_variables(
                            (
                                (
                                    self._input_channels
                                    if i == 0
                                    else self._output_channels
                                ),
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

    @handle_nestable
    def _forward(self, inputs, initial_state=None):
        """
        Perform forward pass of the LSTM layer.

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


# Pooling #
# --------#


class MaxPool2D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        data_format="NHWC",
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Max Pooling over a mini-batch of inputs.

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._data_format = data_format
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.max_pool2d(
            inputs,
            self._kernel_size,
            self._stride,
            self._padding,
            data_format=self._data_format,
        )


class AvgPool2D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        data_format="NHWC",
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Average Pooling over a mini-batch of inputs.

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._data_format = data_format
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.avg_pool2d(
            inputs,
            self._kernel_size,
            self._stride,
            self._padding,
            data_format=self._data_format,
        )


class MaxPool1D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        data_format="NWC",
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Max Pooling over a mini-batch of inputs.

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._data_format = data_format
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.max_pool1d(
            inputs,
            self._kernel_size,
            self._stride,
            self._padding,
            data_format=self._data_format,
        )


class MaxPool3D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        data_format="NDHWC",
        device=None,
        dtype=None,
    ):
        """
        Class for applying 3D Max Pooling over 5D inputs.

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window.
        padding
            Implicit zero padding to be added on both sides.
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._data_format = data_format
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.max_pool3d(
            x,
            self._kernel_size,
            self._stride,
            self._padding,
            data_format=self._data_format,
        )


class AvgPool3D(Module):
    def __init__(
        self,
        kernel_size,
        strides,
        padding,
        /,
        *,
        data_format="NDHWC",
        count_include_pad=False,
        ceil_mode=False,
        divisor_override=None,
    ):
        """
        Class for applying Average Pooling over a mini-batch of inputs.

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If specified, it will be used as divisor,
            otherwise kernel_size will be used. # noqa: E501
        """
        self._kernel_size = kernel_size
        self._stride = strides
        self._padding = padding
        self._data_format = data_format
        self._count_include_pad = count_include_pad
        self._ceil_mode = ceil_mode
        self._divisor_override = divisor_override
        Module.__init__(self)

    def _forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array to the layer.

        Returns
        -------
            The output array of the layer.
        """
        return ivy.avg_pool3d(
            x,
            self._kernel_size,
            self._stride,
            self._padding,
            data_format=self._data_format,
            count_include_pad=self._count_include_pad,
            ceil_mode=self._ceil_mode,
            divisor_override=self._divisor_override,
        )


class AdaptiveAvgPool2d(Module):
    def __init__(
        self,
        output_size,
        device=None,
        dtype=None,
    ):
        """
        Class for applying a 2D adaptive average pooling over mini-batch of inputs.

        Parameters
        ----------
        output_size
            the target output size of the image.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._output_size = output_size
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array to the layer.

        Returns
        -------
            The output array of the layer.
        """
        # TODO: test again once adaptive_avg_pool2d is
        #  implemented for the missing backends.
        return ivy.adaptive_avg_pool2d(
            x,
            self._output_size,
        )


class AdaptiveAvgPool1d(Module):
    def __init__(
        self,
        output_size,
        device=None,
        dtype=None,
    ):
        # TODO: add data_format param
        """
        Class for applying a 1D adaptive average pooling over mini-batch of inputs.

        Parameters
        ----------
        output_size
            An integer or tuple/list of a single integer
            specifying new size of output channels.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._output_size = output_size
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array to the layer.

        Returns
        -------
            The output array of the layer.
        """
        # TODO: test again once adaptive_avg_pool2d is
        #  implemented for the missing backends.
        return ivy.adaptive_avg_pool1d(
            x,
            self._output_size,
        )


class FFT(Module):
    def __init__(
        self,
        dim,
        /,
        *,
        norm="backward",
        n=None,
        out=None,
        device=None,
        dtype=None,
    ):
        """
        Class for applying FFT to input.

        Parameters
        ----------
        dim : int
            Dimension along which to take the FFT.
        norm : str
            Normalization mode. Default: 'backward'
        n : int
            Size of the FFT. Default: None
        out : int
            Size of the output. Default: None
        """
        self._dim = dim
        self._norm = norm
        self._n = n
        self._out = out
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : array
            Input array to take the FFT of.

        Returns
        -------
        array
            The output array of the layer.
        """
        return ivy.fft(
            inputs,
            self._dim,
            norm=self._norm,
            n=self._n,
            out=self._out,
        )


class AvgPool1D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        data_format="NWC",
    ):
        """
        Class for applying Average Pooling over a mini-batch of inputs.

        Parameters
        ----------
        kernel_size
            The size of the window to take an average over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        data_format
            "NCW" or "NWC". Defaults to "NWC".
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._data_format = data_format
        Module.__init__(self)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.avg_pool1d(
            inputs,
            self._kernel_size,
            self._stride,
            self._padding,
            data_format=self._data_format,
        )


class Dct(Module):
    def __init__(
        self,
        *,
        type=2,
        n=None,
        axis=-1,
        norm=None,
        device=None,
        dtype=None,
    ):
        """
        Class for applying the Discrete Cosine Transform over mini-batch of inputs.

        Parameters
        ----------
        x
            The input signal.
        type
            The type of the dct. Must be 1, 2, 3 or 4.
        n
            The length of the transform. If n is less than the input signal lenght,
            then x is truncated, if n is larger then x is zero-padded.
        axis
            The axis to compute the DCT along.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self.type = type
        self.n = n
        self.axis = axis
        self.norm = norm
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array to the layer.

        Returns
        -------
            The output array of the layer.
        """
        return ivy.dct(
            x,
            type=self.type,
            n=self.n,
            axis=self.axis,
            norm=self.norm,
        )


class Embedding(Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        /,
        *,
        max_norm=None,
        weight_initializer=RandomNormal(0.0, 1.0),
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for embedding indices into a dense representation. The Embedding layer is
        a simple lookup table for dense vectors. It's typically used to store word
        embeddings and query them using indices.

        Parameters
        ----------
        input_dims
            Size of the vocabulary.
        output_dims
            Dimension of each embedding vector.
        max_norm
            If given, each embedding vector with L2 norm larger than max_norm is renormalized to have norm max_norm.
        """
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._max_norm = max_norm
        self._w_init = weight_initializer
        self._w_shape = (self._input_dims, self._output_dims)
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        v = {
            "weight": self._w_init.create_variables(
                var_shape=self._w_shape,
                device=device,
                dtype=dtype,
            )
        }
        return v

    def _forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array to the layer.

        Returns
        -------
            The output array of the layer.
        """
        return ivy.embedding(
            self.v.weight,
            x,
            max_norm=self._max_norm,
        )


class Identity(Module):
    """
    The Identity layer is argument insensitive and returns the input argument as output
    when called.

    It's typically used as a placeholder when no operation is to be
    performed. It doesn't have any learnable parameter.
    """

    def __init__(self, dtype=None):
        Module.__init__(self, device=None, v=None, dtype=dtype)

    def _forward(self, x, dtype=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input array.
        dtype
            The desired data type of the internal variables to be created if not
            provided. Default is ``None``.

        Returns
        -------
            The input array as it is.
        """
        return x
