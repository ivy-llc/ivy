"""Collection of Ivy neural network layers as stateful classes."""
# flake8: noqa
# local
import ivy
from ivy.func_wrapper import handle_nestable
from ivy.stateful.initializers import GlorotUniform, Zeros
from ivy.stateful.module import Module
from ivy.stateful.norms import LayerNorm
import copy
from typing import Optional, Any, Union, Callable
import warnings
from ivy import random_uniform as xavier_uniform_

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


# EMBEDDING #
# ----------#


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        /,
        *,
        weight_initializer=GlorotUniform(),
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
        num_embeddingss : int
            Number of embeddings.
        embedding_dim : int
            Dimension of the embeddings.
        padding_idx : int
            If given, pads the output with zeros whenever it encounters the index.
        max_norm : float
            If given, each embedding vector with L2 norm larger than max_norm is renormalized to have norm max_norm.
        weight_initializer : Initializer
            Initializer for the weights.
        device : str
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        v : dict
            the variables for the embedding layer, as a container, constructed internally
            by default.
        dtype
            the desired data type of the internal variables to be created if not
             provided. Default is ``None``.
        """
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._padding_idx = padding_idx
        self._max_norm = max_norm
        self._weight_initializer = weight_initializer
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
            "w": self._weight_initializer.create_variables(
                (self._num_embeddings, self._embedding_dim),
                device,
                self._embedding_dim,
                self._num_embeddings,
                dtype=dtype,
            )
        }
        return v

    def _pad_embd(self, indices, embd):
        mask = ivy.expand_dims(indices == self._padding_idx, axis=-1)
        mask_val = ivy.array(0.0, dtype=embd.dtype)
        return ivy.where(mask, mask_val, embd)

    def _forward(self, indices):
        """
        Forward pass of the layer.

        Parameters
        ----------
        indices
            The input array to the layer.

        Returns
        -------
            The output array of the layer.
        """
        emb = ivy.embedding(self.v.w, indices, max_norm=self._max_norm)
        if self._padding_idx is not None:
            emb = self._pad_embd(indices, emb)
        return emb


class Identity(Module):
    def __init__(self):
        """
        Identity layer. The layer is argument insensitive and returns the input argument
        as output when called.

        It's typically used as a placeholder when no operation is to be
        performed. It doesn't have any learnable parameter.
        """
        Module.__init__(self)

    def _forward(self, x):
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


# Transformer #
# ----------#


def _generate_square_subsequent_mask(
    sz: int,
    device=ivy.default_device(),
    dtype=ivy.default_dtype(),
) -> ivy.NativeArray:
    """
    Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked
    positions are filled with float(0.0).
    """
    return ivy.triu(
        ivy.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: ivy.NativeArray, batch_first: bool) -> Optional[int]:
    # FIXME: It seems class NativeArray hasn't been defined at the time this code is being written, at the time of writing, it will need to have 'is_nested' and 'size()' attributes, or this code will have to be changed based on the latest implementation of NativeArray class
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _canonical_mask(
    mask: Optional[Union[ivy.Array, bool]],
    mask_name: str,
    other_type: Optional[type],
    other_name: str,
    target_type: type,
    check_other: bool = True,
) -> Optional[ivy.Array]:
    if mask is not None:
        mask = ivy.array(mask, dtype=bool)
        if mask.dtype != bool:
            raise AssertionError(f"only bool types of {mask_name} are supported")
        if check_other and other_type is not None:
            if mask.dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        mask = mask.astype(target_type)
        mask = ivy.where(mask, float("-inf"), 0.0)
    return mask


def _none_or_dtype(input: Optional[ivy.NativeArray]) -> Optional[int]:
    if input is None:
        return None
    elif isinstance(input, ivy.NativeArray):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or ivy.NativeArray")


class Transformer(Module):
    """
    A transformer model. User is able to modify the attributes as needed. The
    architecture is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam
    Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,
    and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural
    Information Processing Systems, pages 6000-6010.

    Parameters
    ----------
    d_model
        the number of expected features in the encoder/decoder inputs (default=512).
    nhead
        the number of heads in the multiheadattention models (default=8).
    num_encoder_layers
        the number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers
        the number of sub-decoder-layers in the decoder (default=6).
    dim_feedforward
        the dimension of the feedforward network model (default=2048).
    dropout
        the dropout value (default=0.1).
    activation
        the activation function of encoder/decoder intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    custom_encoder
        custom encoder (default=None).
    custom_decoder
        custom decoder (default=None).
    layer_norm_eps
        the eps value in layer normalization components (default=1e-5).
    batch_first
        If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    norm_first
        if ``True``, encoder and decoder layers will perform LayerNorms before
        other attention and feedforward operations, otherwise after. Default: ``False`` (after).
    bias
        If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
        bias. Default: ``True``.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[ivy.NativeArray], ivy.NativeArray]] = ivy.relu,
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            encoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            decoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def _forward(
        self,
        src: ivy.NativeArray,
        tgt: ivy.NativeArray,
        src_mask: Optional[ivy.NativeArray] = None,
        tgt_mask: Optional[ivy.NativeArray] = None,
        memory_mask: Optional[ivy.NativeArray] = None,
        src_key_padding_mask: Optional[ivy.NativeArray] = None,
        tgt_key_padding_mask: Optional[ivy.NativeArray] = None,
        memory_key_padding_mask: Optional[ivy.NativeArray] = None,
        src_is_causal: Optional[bool] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> ivy.NativeArray:
        r"""
        Take in and process masked source/target sequences.

        Parameters
        ----------
        src
            the sequence to the encoder (required).
        tgt
            the sequence to the decoder (required).
        src_mask
            the additive mask for the src sequence (optional).
        tgt_mask
            the additive mask for the tgt sequence (optional).
        memory_mask
            the additive mask for the encoder output (optional).
        src_key_padding_mask
            the NativeArray mask for src keys per batch (optional).
        tgt_key_padding_mask
            the ivy.NativeArray mask for tgt keys per batch (optional).
        memory_key_padding_mask
            the ivy.NativeArray mask for memory keys per batch (optional).
        src_is_causal
            If specified, applies a causal mask as ``src_mask``.
            Default: ``None``; try to detect a causal mask.
            Warning: ``src_is_causal`` provides a hint that ``src_mask`` is
            the causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.
        tgt_is_causal
            If specified, applies a causal mask as ``tgt_mask``.
            Default: ``None``; try to detect a causal mask.
            Warning: ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
            the causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.
        memory_is_causal
            If specified, applies a causal mask as
            ``memory_mask``.
            Default: ``False``.
            Warning: ``memory_is_causal`` provides a hint that
            ``memory_mask`` is the causal mask. Providing incorrect
            hints can result in incorrect execution, including
            forward and backward compatibility.

        Shape
        -----
        - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
          `(N, S, E)` if `batch_first=True`.
        - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
          `(N, T, E)` if `batch_first=True`.
        - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
        - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
        - memory_mask: :math:`(T, S)`.
        - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
        - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
        - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
        positions. If a BoolTensor is provided, positions with ``True``
        are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
        is provided, it will be added to the attention weight.
        [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
        the attention. If a BoolTensor is provided, the positions with the
        value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

        - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
          `(N, T, E)` if `batch_first=True`.

        Note: Due to the multi-head attention architecture in the transformer model,
        the output sequence length of a transformer is same as the input sequence
        (i.e. target) length of the decoder.

        where S is the source sequence length, T is the target sequence length, N is the
        batch size, E is the feature number
        """
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )
        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device=ivy.default_device(),
        dtype=ivy.default_dtype(),
    ) -> ivy.NativeArray:
        r"""
        Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked
        positions are filled with float(0.0).
        """
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""
    TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args
    ----
    encoder_layer
        an instance of the TransformerEncoderLayer() class (required).
    num_layers
        the number of sub-encoder-layers in the encoder (required).
    norm
        the layer normalization component (optional).
    enable_nested_tensor
        if True, input will automatically convert to nested tensor
        (and convert back on output). This will improve the overall performance of
        TransformerEncoder when padding rate is high. Default: ``True`` (enabled).
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ""
        if not isinstance(encoder_layer, TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn.batch_first was not True"
                + "(use batch_first for better inference performance)"
            )
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.activation_relu_or_gelu was not True"
            )
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps):
            why_not_sparsity_fast_path = (
                f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
            )
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                "enable_nested_tensor is True, but self.use_nested_tensor is False"
                f" because {why_not_sparsity_fast_path}"
            )
            self.use_nested_tensor = False

    def _forward(
        self,
        src: ivy.NativeArray,
        mask: Optional[ivy.NativeArray] = None,
        src_key_padding_mask: Optional[ivy.NativeArray] = None,
        is_causal: Optional[bool] = None,
    ) -> ivy.NativeArray:
        r"""
        Pass the input through the encoder layers in turn.

        Args
        ----
        src
            the sequence to the encoder (required).
        mask
            the mask for the src sequence (optional).
        src_key_padding_mask
            the mask for the src keys per batch (optional).
        is_causal
            If specified, applies a causal mask as ``mask``.
            Default: ``None``; try to detect a causal mask.
            Warning: ``is_causal`` provides a hint that ``mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

        Shape
        -----
        see the docs in Transformer class.
        """
        src_key_padding_mask = _canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = _canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        # FIXME: '_nested_tensor_from_mask_left_aligned' function had only one definition inside a pyi file, which had no body
        # elif (((not hasattr(self, "mask_check")) or self.mask_check)
        #         and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
        #     why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        # FIXME: Requires lower-level implementation
        # elif torch.is_autocast_enabled():
        #     why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda"]
            # FIXME: Requires lower-level implementation
            # if torch.overrides.has_torch_function(tensor_args):
            #     why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            if src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            # FIXME: 'is_grad_enabled' function had only one definition inside a pyi file, which had no body
            # elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
            #     why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
            #                                   "input/output projection weights or biases requires_grad")

            # FIXME: '_nested_tensor_from_mask' function had only one definition inside a pyi file, which had no body
            # if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
            #     convert_to_nested = True
            #     output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
            #     src_key_padding_mask_for_layers = None
        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )
        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(Module):
    r"""
    TransformerDecoder is a stack of N decoder layers.

    Args
    ----
    decoder_layer
        an instance of the TransformerDecoderLayer() class (required).
    num_layers
        the number of sub-decoder-layers in the decoder (required).
    norm
        the layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def _forward(
        self,
        tgt: ivy.NativeArray,
        memory: ivy.NativeArray,
        tgt_mask: Optional[ivy.NativeArray] = None,
        memory_mask: Optional[ivy.NativeArray] = None,
        tgt_key_padding_mask: Optional[ivy.NativeArray] = None,
        memory_key_padding_mask: Optional[ivy.NativeArray] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> ivy.NativeArray:
        r"""
        Pass the inputs (and mask) through the decoder layer in turn.

        Args
        ----
        tgt
            the sequence to the decoder (required).
        memory
            the sequence from the last layer of the encoder (required).
        tgt_mask
            the mask for the tgt sequence (optional).
        memory_mask
            the mask for the memory sequence (optional).
        tgt_key_padding_mask
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask
            the mask for the memory keys per batch (optional).
        tgt_is_causal
            If specified, applies a causal mask as ``tgt mask``.
            Default: ``None``; try to detect a causal mask.
            Warning: ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
            the causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.
        memory_is_causal
            If specified, applies a causal mask as
            ``memory mask``.
            Default: ``False``.
            Warning: ``memory_is_causal`` provides a hint that
            ``memory_mask`` is the causal mask. Providing incorrect
            hints can result in incorrect execution, including
            forward and backward compatibility.

        Shape
        -----
        see the docs in Transformer class.
        """
        output = tgt
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(Module):
    r"""
    TransformerEncoderLayer is made up of self-attn and feedforward network. This
    standard encoder layer is based on the paper "Attention Is All You Need". Ashish
    Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or
    implement in a different way during application.

    TransformerEncoderLayer can handle either traditional ivy.NativeArray inputs,
    or Nested NativeArray inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested NativeArray is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both ivy.NativeArray and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer.

    Args
    ----
    d_model
        the number of expected features in the input (required).
    nhead
        the number of heads in the multiheadattention models (required).
    dim_feedforward
        the dimension of the feedforward network model (default=2048).
    dropout
        the dropout value (default=0.1).
    activation
        the activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    layer_norm_eps
        the eps value in layer normalization components (default=1e-5).
    batch_first
        If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    norm_first
        if ``True``, layer norm is done prior to attention and feedforward
        operations, respectively. Otherwise it's done after. Default: ``False`` (after).
    bias
        If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
        bias. Default: ``True``.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[ivy.NativeArray], ivy.NativeArray]] = ivy.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        if activation is ivy.relu or isinstance(
            activation, ivy.stateful.activations.ReLU
        ):
            self.activation_relu_or_gelu = 1
        elif activation is ivy.gelu or isinstance(
            activation, ivy.stateful.activations.GELU
        ):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = ivy.relu

    def _forward(
        self,
        src: ivy.NativeArray,
        src_mask: Optional[ivy.NativeArray] = None,
        src_key_padding_mask: Optional[ivy.NativeArray] = None,
        is_causal: bool = False,
    ) -> ivy.NativeArray:
        r"""
        Pass the input through the encoder layer.

        Args
        ----
        src
            the sequence to the encoder layer (required).
        src_mask
            the mask for the src sequence (optional).
        src_key_padding_mask
            the mask for the src keys per batch (optional).
        is_causal
            If specified, applies a causal mask as ``src mask``.
            Default: ``False``.
            Warning: ``is_causal`` provides a hint that ``src_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

        Shape
        -----
        see the docs in Transformer class.
        """
        src_key_padding_mask = _canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = _canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ""
        if not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = (
                "neither src_key_padding_mask nor src_mask are not supported with"
                " NestedTensor input"
            )
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        # FIXME: Requires lower-level implementation
        # elif torch.is_autocast_enabled():
        #     why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda"]
            # FIXME: Requires lower-level implementation
            # if torch.overrides.has_torch_function(tensor_args):
            #     why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            if not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "some ivy.NativeArray argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            # FIXME: 'is_grad_enabled' function had only one definition inside a pyi file, which had no body
            # elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
            #     why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
            #                                   "input/output projection weights or biases requires_grad")
            # FIXME: '_transformer_encoder_layer_fwd' function had only one definition inside a pyi file, which had no body
            # if not why_not_sparsity_fast_path:
            #     merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
            #     return torch._transformer_encoder_layer_fwd(
            #         src,
            #         self.self_attn.embed_dim,
            #         self.self_attn.num_heads,
            #         self.self_attn.in_proj_weight,
            #         self.self_attn.in_proj_bias,
            #         self.self_attn.out_proj.weight,
            #         self.self_attn.out_proj.bias,
            #         self.activation_relu_or_gelu == 2,
            #         self.norm_first,
            #         self.norm1.eps,
            #         self.norm1.weight,
            #         self.norm1.bias,
            #         self.norm2.weight,
            #         self.norm2.bias,
            #         self.linear1.weight,
            #         self.linear1.bias,
            #         self.linear2.weight,
            #         self.linear2.bias,
            #         merged_mask,
            #         mask_type,
            #     )
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: ivy.NativeArray,
        attn_mask: Optional[ivy.NativeArray],
        key_padding_mask: Optional[ivy.NativeArray],
        is_causal: bool = False,
    ) -> ivy.NativeArray:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: ivy.NativeArray) -> ivy.NativeArray:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    r"""
    TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward
    network. This standard decoder layer is based on the paper "Attention Is All You
    Need". Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need.
    In Advances in Neural Information Processing Systems, pages 6000-6010. Users may
    modify or implement in a different way during application.

    Args
    ----
    d_model
        the number of expected features in the input (required).
    nhead
        the number of heads in the multiheadattention models (required).
    dim_feedforward
        the dimension of the feedforward network model (default=2048).
    dropout
        the dropout value (default=0.1).
    activation
        the activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    layer_norm_eps
        the eps value in layer normalization components (default=1e-5).
    batch_first
        If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    norm_first
        if ``True``, layer norm is done prior to self attention, multihead
        attention and feedforward operations, respectively. Otherwise it's done after.
        Default: ``False`` (after).
    bias
        If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
        bias. Default: ``True``.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[ivy.NativeArray], ivy.NativeArray]] = ivy.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = ivy.relu
        super().__setstate__(state)

    def _forward(
        self,
        tgt: ivy.NativeArray,
        memory: ivy.NativeArray,
        tgt_mask: Optional[ivy.NativeArray] = None,
        memory_mask: Optional[ivy.NativeArray] = None,
        tgt_key_padding_mask: Optional[ivy.NativeArray] = None,
        memory_key_padding_mask: Optional[ivy.NativeArray] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> ivy.NativeArray:
        r"""
        Pass the inputs (and mask) through the decoder layer.

        Args
        ----
        tgt
            the sequence to the decoder layer (required).
        memory
            the sequence from the last layer of the encoder (required).
        tgt_mask
            the mask for the tgt sequence (optional).
        memory_mask
            the mask for the memory sequence (optional).
        tgt_key_padding_mask
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask
            the mask for the memory keys per batch (optional).
        tgt_is_causal
            If specified, applies a causal mask as ``tgt mask``.
            Default: ``False``.
            Warning: ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
            the causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.
        memory_is_causal
            If specified, applies a causal mask as
            ``memory mask``.
            Default: ``False``.
            Warning: ``memory_is_causal`` provides a hint that
            ``memory_mask`` is the causal mask. Providing incorrect
            hints can result in incorrect execution, including
            forward and backward compatibility.

        Shape
        -----
        see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: ivy.NativeArray,
        attn_mask: Optional[ivy.NativeArray],
        key_padding_mask: Optional[ivy.NativeArray],
        is_causal: bool = False,
    ) -> ivy.NativeArray:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: ivy.NativeArray,
        mem: ivy.NativeArray,
        attn_mask: Optional[ivy.NativeArray],
        key_padding_mask: Optional[ivy.NativeArray],
        is_causal: bool = False,
    ) -> ivy.NativeArray:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x: ivy.NativeArray) -> ivy.NativeArray:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return [copy.deepcopy(module) for i in range(N)]


def _get_activation_fn(activation: str) -> Callable[[ivy.NativeArray], ivy.NativeArray]:
    if activation == "relu":
        return ivy.relu
    elif activation == "gelu":
        return ivy.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
    mask: Optional[ivy.NativeArray],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """
    Return whether the given attention mask is causal.

    Warning
    -------
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True
    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False
    return make_causal
