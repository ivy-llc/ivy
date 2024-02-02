###############################################################################
#
# This script includes the source code of the perceiverIO as released by
# DeepMind under the Apache-2.0 license. The original repo can be found at:
# https://github.com/deepmind/deepmind-research/tree/master/perceiver
#
###############################################################################

import abc
import jax
import math
import einops
import functools
import haiku as hk
import numpy as np
import jax.numpy as jnp

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

###############################################################################
#
# position_encoding.py
#
###############################################################################


def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False
):
    """Generate a Fourier frequency position encoding with linear spacing.

    Args:
      pos: The position of n points in d dimensional space.
        A jnp array of shape [n, d].
      num_bands: The number of bands (K) to use.
      max_resolution: The maximum resolution (i.e. the number of pixels per dim).
        A tuple representing resolution for each dimension
      concat_pos: Concatenate the input position encoding to the Fourier features?
      sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
        frequency band.
    Returns:
      embedding: A 1D jnp array of shape [n, n_channels]. If concat_pos is True
        and sine_only is False, output dimensions are ordered as:
          [dim_1, dim_2, ..., dim_d,
           sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
           sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
           cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
           cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
         where dim_i is pos[:, i] and f_k is the kth frequency band.
    """
    min_freq = 1.0
    # Nyquist frequency at the target resolution:

    freq_bands = jnp.stack(
        [
            jnp.linspace(min_freq, res / 2, num=num_bands, endpoint=True)
            for res in max_resolution
        ],
        axis=0,
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
    per_pos_features = jnp.reshape(
        per_pos_features, [-1, np.prod(per_pos_features.shape[1:])]
    )

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = jnp.sin(jnp.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = jnp.concatenate(
            [jnp.sin(jnp.pi * per_pos_features), jnp.cos(jnp.pi * per_pos_features)],
            axis=-1,
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = jnp.concatenate([pos, per_pos_features], axis=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """Generate an array of position indices for an N-D input array.

    Args:
      index_dims: The shape of the index dimensions of the input array.
      output_range: The min and max values taken by each input index dimension.
    Returns:
      A jnp array of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
    """

    def _linspace(n_xels_per_dim):
        return jnp.linspace(
            output_range[0],
            output_range[1],
            num=n_xels_per_dim,
            endpoint=True,
            dtype=jnp.float32,
        )

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = jnp.meshgrid(*dim_ranges, indexing="ij")

    return jnp.stack(array_index_grid, axis=-1)


class AbstractPositionEncoding(hk.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def __call__(self, batch_size, pos):
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dim, num_channels=128, init_scale=0.02, name=None):
        super(TrainablePositionEncoding, self).__init__(name=name)
        self._index_dim = index_dim
        self._num_channels = num_channels
        self._init_scale = init_scale

    def __call__(self, batch_size, pos=None):
        del pos  # Unused.
        pos_embs = hk.get_parameter(
            "pos_embs",
            [self._index_dim, self._num_channels],
            init=hk.initializers.TruncatedNormal(stddev=self._init_scale),
        )

        if batch_size is not None:
            pos_embs = jnp.broadcast_to(
                pos_embs[None, :, :], (batch_size,) + pos_embs.shape
            )
        return pos_embs


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """Checks or builds spatial position features (x, y, ...).

    Args:
      pos: None, or an array of position features. If None, position features
        are built. Otherwise, their size is checked.
      index_dims: An iterable giving the spatial/index size of the data to be
        featurized.
      batch_size: The batch size of the data to be featurized.
    Returns:
      An array of position features, of shape [batch_size, prod(index_dims)].
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = jnp.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = jnp.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        assert pos.shape[-1] == len(index_dims)

    return pos


class FourierPositionEncoding(AbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(
        self,
        index_dims,
        num_bands,
        concat_pos=True,
        max_resolution=None,
        sine_only=False,
        name=None,
    ):
        super(FourierPositionEncoding, self).__init__(name=name)
        self._num_bands = num_bands
        self._concat_pos = concat_pos
        self._sine_only = sine_only
        self._index_dims = index_dims
        # Use the index dims as the maximum resolution if it's not provided.
        self._max_resolution = max_resolution or index_dims

    def __call__(self, batch_size, pos=None):
        pos = _check_or_build_spatial_positions(pos, self._index_dims, batch_size)
        build_ff_fn = functools.partial(
            generate_fourier_features,
            num_bands=self._num_bands,
            max_resolution=self._max_resolution,
            concat_pos=self._concat_pos,
            sine_only=self._sine_only,
        )
        return jax.vmap(build_ff_fn, 0, 0)(pos)


class PositionEncodingProjector(AbstractPositionEncoding):
    """Projects a position encoding to a target size."""

    def __init__(self, output_size, base_position_encoding, name=None):
        super(PositionEncodingProjector, self).__init__(name=name)
        self._output_size = output_size
        self._base_position_encoding = base_position_encoding

    def __call__(self, batch_size, pos=None):
        base_pos = self._base_position_encoding(batch_size, pos)
        projected_pos = hk.Linear(output_size=self._output_size)(base_pos)
        return projected_pos


def build_position_encoding(
    position_encoding_type,
    index_dims,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
    name=None,
):
    """Builds the position encoding."""

    if position_encoding_type == "trainable":
        assert trainable_position_encoding_kwargs is not None
        output_pos_enc = TrainablePositionEncoding(
            # Construct 1D features:
            index_dim=np.prod(index_dims),
            name=name,
            **trainable_position_encoding_kwargs,
        )
    elif position_encoding_type == "fourier":
        assert fourier_position_encoding_kwargs is not None
        output_pos_enc = FourierPositionEncoding(
            index_dims=index_dims, name=name, **fourier_position_encoding_kwargs
        )
    else:
        raise ValueError(f"Unknown position encoding: {position_encoding_type}.")

    if project_pos_dim > 0:
        # Project the position encoding to a target dimension:
        output_pos_enc = PositionEncodingProjector(
            output_size=project_pos_dim, base_position_encoding=output_pos_enc
        )

    return output_pos_enc


###############################################################################
#
# io_processors.py
#
###############################################################################

ModalitySizeT = Mapping[str, int]
PreprocessorOutputT = Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]
PreprocessorT = Callable[..., PreprocessorOutputT]
PostprocessorT = Callable[..., Any]


def reverse_space_to_depth(
    frames: jnp.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> jnp.ndarray:
    """Reverse space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b h w (dh dw c) -> b (h dh) (w dw) c",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


def space_to_depth(
    frames: jnp.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> jnp.ndarray:
    """Space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b (h dh) (w dw) c -> b h w (dh dw c)",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b (t dt) (h dh) (w dw) c -> b t h w (dt dh dw c)",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


def extract_patches(
    images: jnp.ndarray,
    sizes: Sequence[int],
    strides: Sequence[int],
    rates: Sequence[int],
    padding: str = "VALID",
) -> jnp.ndarray:
    """Extract patches from images.

    This function is a wrapper for jax.lax.conv_general_dilated_patches
    to conforms to the same interface as tf.image.extract_patches.
    The function extracts patches of shape sizes from the input images in the same
    manner as a convolution with kernel of shape sizes, stride equal to strides,
    and the given padding scheme.
    The patches are stacked in the channel dimension.

    Args:
      images: input batch of images of shape [B, H, W, C].
      sizes: size of extracted patches. Must be [1, size_rows, size_cols, 1].
      strides: strides, must be [1, stride_rows, stride_cols, 1].
      rates: sampling rate (as in dilated convolutions),
        must be [1, rate_rows, rate_cols, 1].
      padding: padding algorithm to use.
    Returns:
      Tensor of shape [B, patch_rows, patch_cols, size_rows * size_cols * C]
    """

    if len(sizes) != 4 or sizes[0] != 1 or sizes[3] != 1:
        raise ValueError(
            f"Shape of sizes must be [1, size_rows, size_cols, 1], got {sizes}."
        )
    if len(strides) != 4 or strides[0] != 1 or strides[3] != 1:
        raise ValueError(
            f"Shape of strides must be [1, size_rows, size_cols, 1], " f"got {strides}."
        )
    if len(rates) != 4 or rates[0] != 1 or rates[3] != 1:
        raise ValueError(
            f"Shape of rates must be [1, size_rows, size_cols, 1], got {rates}."
        )
    if images.ndim != 4:
        raise ValueError(
            f"Rank of images must be 4 (got tensor of shape {jnp.shape(images)})"
        )
    # Rearrange axes of images to NCHW for conv_general_dilated_patches
    images = einops.rearrange(images, "n h w c -> n c h w")
    channels = images.shape[1]
    patches = jax.lax.conv_general_dilated_patches(
        images, sizes[1:-1], strides[1:-1], padding, rhs_dilation=rates[1:-1]
    )
    # conv_general_dilated_patches returns patches in channel-major order.
    # Rearrange to match interface of tf.image.extract_patches.
    patches = einops.rearrange(
        patches,
        "n (c ph pw) h w -> n h w (ph pw c)",
        c=channels,
        ph=sizes[1],
        pw=sizes[2],
    )
    return patches


def patches_for_flow(inputs: jnp.ndarray) -> jnp.ndarray:
    """Extract 3x3x2 image patches for flow inputs."""

    def pad_and_extract_patches(inputs):
        padded_inputs = jnp.pad(
            inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="constant"
        )
        return extract_patches(
            padded_inputs,
            sizes=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            rates=[1, 1, 1, 1],
        )

    return jax.vmap(pad_and_extract_patches, in_axes=1, out_axes=1)(inputs)


#  ------------------------------------------------------------
#  -------------------  Up/down-sampling  ---------------------
#  ------------------------------------------------------------


class Conv2DDownsample(hk.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(
        self,
        num_layers: int = 1,
        num_channels: int = 64,
        use_batchnorm: bool = True,
        bn_config: Optional[Mapping[str, float]] = None,
        name: Optional[str] = None,
    ):
        """Constructs a Conv2DDownsample model.

        Args:
          num_layers: The number of conv->max_pool layers.
          num_channels: The number of conv output channels.
          use_batchnorm: Whether to use batchnorm.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers. By default the
            ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
          name: Name of the module.
        """
        super().__init__(name=name)

        self._num_layers = num_layers
        self._use_batchnorm = use_batchnorm

        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        self.layers = []
        for _ in range(self._num_layers):
            conv = hk.Conv2D(
                output_channels=num_channels,
                kernel_shape=7,
                stride=2,
                with_bias=False,
                padding="SAME",
                name="conv",
            )
            if use_batchnorm:
                batchnorm = hk.BatchNorm(name="batchnorm", **bn_config)
            else:
                batchnorm = None
            self.layers.append(dict(conv=conv, batchnorm=batchnorm))

    def __call__(
        self, inputs: jnp.ndarray, *, is_training: bool, test_local_stats: bool = False
    ) -> jnp.ndarray:
        out = inputs
        for layer in self.layers:
            out = layer["conv"](out)
            if layer["batchnorm"] is not None:
                out = layer["batchnorm"](out, is_training, test_local_stats)
            out = jax.nn.relu(out)
            out = hk.max_pool(
                out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
            )
        return out


class Conv2DUpsample(hk.Module):
    """Upsamples 4x using 2 2D transposed convolutions."""

    def __init__(
        self,
        n_outputs: int,
        name: Optional[str] = None,
    ):
        """Constructs a Conv2DUpsample model.

        Args:
          n_outputs: The number of output channels of the module.
          name: Name of the module.
        """
        super().__init__(name=name)

        self.transp_conv1 = hk.Conv2DTranspose(
            output_channels=n_outputs * 2,
            kernel_shape=4,
            stride=2,
            with_bias=True,
            padding="SAME",
            name="transp_conv_1",
        )

        self.transp_conv2 = hk.Conv2DTranspose(
            output_channels=n_outputs,
            kernel_shape=4,
            stride=2,
            with_bias=True,
            padding="SAME",
            name="transp_conv_2",
        )

    def __call__(
        self, inputs: jnp.ndarray, *, is_training: bool, test_local_stats: bool = False
    ) -> jnp.ndarray:
        out = inputs
        out = self.transp_conv1(out)
        out = jax.nn.relu(out)
        out = self.transp_conv2(out)

        return out


class Conv3DUpsample(hk.Module):
    """Simple convolutional auto-encoder."""

    def __init__(
        self,
        n_outputs: int,
        n_time_upsamples: int = 2,
        n_space_upsamples: int = 4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self._n_outputs = n_outputs
        self._n_time_upsamples = n_time_upsamples
        self._n_space_upsamples = n_space_upsamples

    def __call__(self, x: jnp.ndarray, *, is_training: bool) -> jnp.ndarray:
        n_upsamples = max(self._n_time_upsamples, self._n_space_upsamples)

        time_stride = 2
        space_stride = 2

        for i in range(n_upsamples):
            if i >= self._n_time_upsamples:
                time_stride = 1
            if i >= self._n_space_upsamples:
                space_stride = 1

            channels = self._n_outputs * pow(2, n_upsamples - 1 - i)

            x = hk.Conv3DTranspose(
                output_channels=channels,
                stride=[time_stride, space_stride, space_stride],
                kernel_shape=[4, 4, 4],
                name=f"conv3d_transpose_{i}",
            )(x)
            if i != n_upsamples - 1:
                x = jax.nn.relu(x)

        return x


class ImagePreprocessor(hk.Module):
    """Image preprocessing for Perceiver Encoder."""

    def __init__(
        self,
        prep_type="conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        position_encoding_type: str = "fourier",
        n_extra_pos_mlp: int = 0,
        num_channels: int = 64,
        conv_after_patching: bool = False,
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        name: Optional[str] = None,
        **position_encoding_kwargs,
    ):
        super().__init__(name=name)

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError("Invalid prep_type!")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(
                f"Invalid value {concat_or_add_pos} for concat_or_add_pos."
            )

        self._prep_type = prep_type
        self._spatial_downsample = spatial_downsample
        self._temporal_downsample = temporal_downsample
        self._concat_or_add_pos = concat_or_add_pos
        self._conv_after_patching = conv_after_patching
        self._num_channels = num_channels

        if self._prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(
                convnet_num_layers
            )
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial "
                    "and 1 expected for temporal "
                    "downsampling with conv."
                )

            self.convnet = Conv2DDownsample(
                num_layers=int(convnet_num_layers),
                num_channels=num_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )
        elif self._prep_type == "conv1x1":
            assert temporal_downsample == 1, "conv1x1 does not downsample in time."
            self.convnet_1x1 = hk.Conv2D(
                num_channels,
                kernel_shape=[1, 1],
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=[spatial_downsample, spatial_downsample],
            )

        # Partially construct the positional encoding function.
        # We fully construct it when we know the input size.
        self._positional_encoding_ctor = functools.partial(
            build_position_encoding,
            position_encoding_type=position_encoding_type,
            **position_encoding_kwargs,
        )

        # Stack MLPs to get a deeper positional embedding.
        self._n_extra_pos_mlp = n_extra_pos_mlp

    def _build_network_inputs(
        self, inputs: jnp.ndarray, pos: jnp.ndarray, network_input_is_1d: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # Reshape input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = jnp.reshape(inputs, [batch_size, np.prod(index_dims), -1])

        # Construct the position encoding.
        pos_enc = self._positional_encoding_ctor(index_dims=index_dims)(
            batch_size=batch_size, pos=pos
        )

        for i in range(0, self._n_extra_pos_mlp):
            pos_enc += hk.Linear(pos_enc.shape[-1])(pos_enc)
            if i < (self._n_extra_pos_mlp - 1):
                pos_enc = jax.nn.relu(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = jnp.reshape(pos_enc, list(sh)[:-1] + [-1])

        if self._concat_or_add_pos == "concat":
            inputs_with_pos = jnp.concatenate([inputs, pos_enc], axis=-1)
        elif self._concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        network_input_is_1d: bool = True,
    ) -> PreprocessorOutputT:
        if self._prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            conv = self.convnet
            if len(inputs.shape) == 5:
                conv = hk.BatchApply(conv)

            inputs = conv(inputs, is_training=is_training)
        elif self._prep_type == "conv1x1":
            # maps inputs to 64d

            conv = self.convnet_1x1

            if len(inputs.shape) == 5:
                conv = hk.BatchApply(conv)

            inputs = conv(inputs)
        elif self._prep_type == "patches":
            # Space2depth featurization.
            # Video: B x T x H x W x C
            inputs = space_to_depth(
                inputs,
                temporal_block_size=self._temporal_downsample,
                spatial_block_size=self._spatial_downsample,
            )

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # for flow
                inputs = jnp.squeeze(inputs, axis=1)

            if self._conv_after_patching:
                inputs = hk.Linear(self._num_channels, name="patches_linear")(inputs)
        elif self._prep_type == "pixels":
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[
                    :, :: self._spatial_downsample, :: self._spatial_downsample
                ]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :,
                    :: self._temporal_downsample,
                    :: self._spatial_downsample,
                    :: self._spatial_downsample,
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        inputs, inputs_without_pos = self._build_network_inputs(
            inputs, pos, network_input_is_1d
        )
        modality_sizes = None  # Size for each modality, only needed for multimodal
        return inputs, modality_sizes, inputs_without_pos


class ImagePostprocessor(hk.Module):
    """Image postprocessing for Perceiver."""

    def __init__(
        self,
        postproc_type: str = "pixels",
        spatial_upsample: int = 1,
        temporal_upsample: int = 1,
        n_outputs: int = -1,  # only relevant for 'conv1x1', 'conv', and 'raft'
        input_reshape_size: Optional[Sequence[int]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if postproc_type not in ("conv", "patches", "pixels", "raft", "conv1x1"):
            raise ValueError("Invalid postproc_type!")

        # Architecture parameters:
        self._postproc_type = postproc_type

        self._temporal_upsample = temporal_upsample
        self._spatial_upsample = spatial_upsample
        self._input_reshape_size = input_reshape_size

        if self._postproc_type == "pixels":
            # No postprocessing.
            if self._temporal_upsample != 1 or self._spatial_upsample != 1:
                raise ValueError("Pixels postprocessing should not currently upsample.")
        elif self._postproc_type == "conv1x1":
            assert self._temporal_upsample == 1, "conv1x1 does not upsample in time."
            if n_outputs == -1:
                raise ValueError("Expected value for n_outputs")
            self.conv1x1 = hk.Conv2D(
                n_outputs,
                kernel_shape=[1, 1],
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=[self._spatial_upsample, self._spatial_upsample],
            )
        elif self._postproc_type == "conv":
            if n_outputs == -1:
                raise ValueError("Expected value for n_outputs")
            if self._temporal_upsample != 1:

                def int_log2(x):
                    return int(np.round(np.log(x) / np.log(2)))

                self.convnet = Conv3DUpsample(
                    n_outputs, int_log2(temporal_upsample), int_log2(spatial_upsample)
                )
            else:
                self.convnet = Conv2DUpsample(n_outputs)

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        modality_sizes: Optional[ModalitySizeT] = None,
    ) -> jnp.ndarray:
        if self._input_reshape_size is not None:
            inputs = jnp.reshape(
                inputs,
                [inputs.shape[0]] + list(self._input_reshape_size) + [inputs.shape[-1]],
            )

        if self._postproc_type == "conv" or self._postproc_type == "raft":
            # Convnet image featurization.
            conv = self.convnet
            if len(inputs.shape) == 5 and self._temporal_upsample == 1:
                conv = hk.BatchApply(conv)
            inputs = conv(inputs, is_training=is_training)
        elif self._postproc_type == "conv1x1":
            inputs = self.conv1x1(inputs)
        elif self._postproc_type == "patches":
            inputs = reverse_space_to_depth(
                inputs, self._temporal_upsample, self._spatial_upsample
            )

        return inputs


class OneHotPreprocessor(hk.Module):
    """One-hot preprocessor for Perceiver Encoder."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        network_input_is_1d: bool = True,
    ) -> PreprocessorOutputT:
        # Add a dummy index dimension.
        inputs = inputs[:, None, :]

        # No position encodings, so the 1st (input) and 3rd (inputs_without_pos)
        # outputs are identical.
        return inputs, None, inputs


class AudioPreprocessor(hk.Module):
    """Audio preprocessing for Perceiver Encoder."""

    def __init__(
        self,
        prep_type: str = "patches",
        samples_per_patch: int = 96,
        position_encoding_type: str = "fourier",
        n_extra_pos_mlp: int = 0,
        concat_or_add_pos: str = "concat",
        name: Optional[str] = None,
        **position_encoding_kwargs,
    ):
        super().__init__(name=name)

        if prep_type not in ("patches",):
            raise ValueError("Invalid prep_type!")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(
                f"Invalid value {concat_or_add_pos} for concat_or_add_pos."
            )

        self._samples_per_patch = samples_per_patch
        self._concat_or_add_pos = concat_or_add_pos

        # Partially construct the positional encoding function.
        # We fully construct it when we know the input size.
        self._positional_encoding_ctor = functools.partial(
            build_position_encoding,
            position_encoding_type=position_encoding_type,
            **position_encoding_kwargs,
        )

        # for deeper positional embeddings
        self._n_extra_pos_mlp = n_extra_pos_mlp

    def _build_network_inputs(
        self, inputs: jnp.ndarray, pos: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # Construct the position encoding.
        pos_enc = self._positional_encoding_ctor(index_dims=index_dims)(
            batch_size=batch_size, pos=pos
        )

        for i in range(0, self._n_extra_pos_mlp):
            pos_enc += hk.Linear(pos_enc.shape[-1])(pos_enc)
            if i < (self._n_extra_pos_mlp - 1):
                pos_enc = jax.nn.relu(pos_enc)

        if self._concat_or_add_pos == "concat":
            inputs_with_pos = jnp.concatenate([inputs, pos_enc], axis=-1)
        elif self._concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        network_input_is_1d: bool = True,
    ) -> PreprocessorOutputT:
        inputs = jnp.reshape(inputs, [inputs.shape[0], -1, self._samples_per_patch])

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos)
        modality_sizes = None  # Size for each modality, only needed for multimodal
        return inputs, modality_sizes, inputs_without_pos


class AudioPostprocessor(hk.Module):
    """Audio postprocessing for Perceiver."""

    def __init__(
        self,
        postproc_type: str = "patches",  # 'conv', 'patches', 'pixels'
        samples_per_patch: int = 96,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if postproc_type not in ("patches",):
            raise ValueError("Invalid postproc_type!")
        self._samples_per_patch = samples_per_patch

        # Architecture parameters:
        self._postproc_type = postproc_type

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        modality_sizes: Optional[ModalitySizeT] = None,
    ) -> jnp.ndarray:
        out = hk.Linear(self._samples_per_patch)(inputs)
        return jnp.reshape(out, [inputs.shape[0], -1])


class IdentityPostprocessor(hk.Module):
    """Passes through the inputs unchanged."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        modality_sizes: Optional[ModalitySizeT] = None,
    ) -> jnp.ndarray:
        return inputs


def restructure(
    modality_sizes: ModalitySizeT, inputs: jnp.ndarray
) -> Mapping[str, jnp.ndarray]:
    """Partitions a [B, N, C] tensor into tensors for each modality.

    Args:
      modality_sizes: dict specifying the size of the modality
      inputs: input tensor
    Returns:
      dict mapping name of modality to its associated tensor.
    """
    outputs = {}
    index = 0
    # Apply a predictable ordering to the modalities
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


class MultimodalPreprocessor(hk.Module):
    """Multimodal preprocessing for Perceiver Encoder.

    Inputs for each modality is preprocessed then padded with trainable position
    embeddings to have the same number of channels.
    """

    def __init__(
        self,
        modalities: Mapping[str, PreprocessorT],
        mask_probs: Optional[Mapping[str, float]] = None,
        min_padding_size: int = 2,
        name: Optional[str] = None,
    ):
        """Constructor.

        Args:
          modalities: dict mapping modality name to preprocessor
          mask_probs: dict mapping modality name to masking probability of that
            modality
          min_padding_size: the minimum padding size for all modalities.
            The final output will have num_channels equal to the maximum channels
            across all modalities plus min_padding_size.
          name: name of module
        """
        super().__init__(name=name)
        self._modalities = modalities
        self._min_padding_size = min_padding_size
        self._mask_probs = mask_probs

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        network_input_is_1d: bool = True,
    ) -> PreprocessorOutputT:
        outputs = {}
        inputs_without_pos = {}
        for modality, preprocessor in self._modalities.items():
            outputs[modality], _, inputs_without_pos[modality] = preprocessor(
                inputs[modality],
                is_training=is_training,
                pos=pos,
                network_input_is_1d=network_input_is_1d,
            )

        common_channel_size = (
            max(o.shape[2] for o in outputs.values()) + self._min_padding_size
        )

        padded = {}
        modality_sizes = {}
        for modality, output in outputs.items():
            pos_enc = TrainablePositionEncoding(
                1,
                num_channels=common_channel_size - output.shape[2],
                init_scale=0.02,
                name=f"{modality}_padding",
            )
            padding = jnp.broadcast_to(
                pos_enc(batch_size=output.shape[0]),
                [
                    output.shape[0],
                    output.shape[1],
                    common_channel_size - output.shape[2],
                ],
            )
            output_padded = jnp.concatenate([output, padding], axis=2)

            if self._mask_probs is not None:
                # Randomly mask out each token corresponding to this modality
                mask_token = TrainablePositionEncoding(
                    1,
                    num_channels=output_padded.shape[2],
                    init_scale=0.02,
                    name=f"{modality}_mask_token",
                )(output.shape[0])
                mask_prob = self._mask_probs[modality]
                rng = hk.next_rng_key()
                mask = jax.random.bernoulli(
                    rng, mask_prob, shape=[output.shape[0], output.shape[1]]
                )
                mask = jnp.expand_dims(mask, axis=2)
                output_padded = (1 - mask) * output_padded + mask * mask_token

            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape[1]

        # Apply a predictable ordering to the modalities
        padded_ls = [padded[k] for k in sorted(padded.keys())]
        return (jnp.concatenate(padded_ls, axis=1), modality_sizes, inputs_without_pos)


class MultimodalPostprocessor(hk.Module):
    """Multimodal postprocessing for Perceiver."""

    def __init__(
        self,
        modalities: Mapping[str, PostprocessorT],
        input_is_dict: bool = False,
        name: Optional[str] = None,
    ):
        """Constructor.

        Args:
          modalities: dict mapping modality name to post processor for that modality
          input_is_dict: If True, input is assumed to be dictionary structured,
            and outputs keep the same dictionary shape. If False, input is a tensor
            which is sliced up during postprocessing by `modality_sizes`.
          name: name of the module
        """
        super().__init__(name=name)
        self._modalities = modalities
        self._input_is_dict = input_is_dict

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        modality_sizes: Optional[ModalitySizeT] = None,
    ) -> Mapping[str, jnp.ndarray]:
        if not self._input_is_dict:
            # Slice up modalities by their sizes.
            assert modality_sizes is not None
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)
        outputs = {
            modality: postprocessor(
                inputs[modality], is_training=is_training, pos=pos, modality_sizes=None
            )
            for modality, postprocessor in self._modalities.items()
        }
        return outputs


class ClassificationPostprocessor(hk.Module):
    """Classification postprocessing for Perceiver."""

    def __init__(self, num_classes: int, name: Optional[str] = None):
        super().__init__(name=name)
        self._num_classes = num_classes

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        modality_sizes: Optional[ModalitySizeT] = None,
    ) -> jnp.ndarray:
        logits = hk.Linear(self._num_classes)(inputs)
        return logits[:, 0, :]


class ProjectionPostprocessor(hk.Module):
    """Projection postprocessing for Perceiver."""

    def __init__(self, num_outputs: int, name: Optional[str] = None):
        super().__init__(name=name)
        self._num_outputs = num_outputs

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        is_training: bool,
        pos: Optional[jnp.ndarray] = None,
        modality_sizes: Optional[ModalitySizeT] = None,
    ) -> jnp.ndarray:
        logits = hk.Linear(self._num_outputs)(inputs)
        return logits


class EmbeddingDecoder(hk.Module):
    """Haiku module to decode embeddings."""

    def __init__(self, embedding_matrix: jnp.ndarray, name="embedding_decoder"):
        """Constructs the module.

        Args:
          embedding_matrix: Array of shape [vocab_size, d_model].
          name: Name of the module.
        """
        super().__init__(name=name)
        self._embedding_matrix = embedding_matrix
        self._vocab_size, self._d_model = embedding_matrix.shape

    def __call__(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = embeddings.shape
        output = jnp.matmul(
            embeddings.reshape([-1, self._d_model]),  # Flatten batch dim
            jnp.transpose(self._embedding_matrix),
        )
        bias = hk.get_parameter("bias", shape=[self._vocab_size], init=jnp.zeros)
        output = output + bias
        return output.reshape([batch_size, seq_len, self._vocab_size])


###############################################################################
#
# perceiver.py
#
###############################################################################


def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
    """Computes multi-head attention using a query, key and value.

    Args:
      q: Query with shape [batch, q_indices, num_heads, head_dim].
      k: Key with shape [batch, kv_indices, num_heads, head_dim].
      v: Value with shape [batch, kv_indices, num_heads, head_dim].
      dropout_prob: dropout probability on the attention weights.
      attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
        which attentions are valid
    Returns:
      Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    attention = jnp.einsum("bthd,bThd->bhtT", q, k)

    scale = 1.0 / math.sqrt(q_head_dim)
    attention *= scale

    if attention_mask is not None:
        # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
        # left-padded sampling.
        large_k = jnp.array(
            1e4 if attention.dtype == jnp.float16 else 1e30, dtype=attention.dtype
        )

        attention = jnp.where(attention_mask[:, None, :, :], attention, -large_k)

    normalized = jax.nn.softmax(attention)
    if dropout_prob > 0:
        normalized = hk.dropout(hk.next_rng_key(), dropout_prob, normalized)
    summed = jnp.einsum("bhtT,bThd->bthd", normalized, v)
    summed = jnp.reshape(summed, [batch, q_indices, hiddens])

    if attention_mask is not None:
        # If all attended tokens are masked, or for masked tokens
        # some rows of logits gets completely masked, in which case the softmax
        # gives a uniform row and we obtain non-zero outputs where it should be
        # zero. We force zeros.
        wipe_attn = jnp.all(
            attention_mask == 0, axis=2, keepdims=True
        )  # shape (B, T, 1)
        summed = jnp.where(wipe_attn, jnp.zeros_like(summed), summed)
    return summed


def conv_1d(output_channels, init_scale=1.0, with_bias=True, name=None):
    """A 1D convolution."""
    return hk.Linear(
        output_size=output_channels,
        with_bias=with_bias,
        w_init=hk.initializers.VarianceScaling(init_scale),
        name=name,
    )


def layer_norm(x, name=None):
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def make_cross_attention_mask(query_mask, kv_mask):
    batch_size, query_len = query_mask.shape
    _, key_len = kv_mask.shape
    mask = jax.vmap(jnp.outer)(query_mask, kv_mask)
    assert mask.shape == (batch_size, query_len, key_len)
    return mask


#  -----------------------------------------------------------
#  -----------------------  Modules  -------------------------
#  -----------------------------------------------------------


class Attention(hk.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(
        self,
        num_heads=8,
        init_scale=1.0,
        with_final_bias=True,
        final_init_scale_multiplier=1.0,
        dropout_prob=0.0,
        qk_channels=None,
        v_channels=None,
        output_channels=None,
        name=None,
    ):
        super(Attention, self).__init__(name=name)
        self._num_heads = num_heads
        self._init_scale = init_scale
        self._with_final_bias = with_final_bias
        self._final_init_scale = final_init_scale_multiplier * init_scale
        self._dropout_prob = dropout_prob

        # If none of these are passed, the Q input determines the output shape:
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._output_channels = output_channels

    def __call__(self, inputs_q, inputs_kv, attention_mask=None):
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if self._qk_channels is None:
            self._qk_channels = inputs_q.shape[-1]
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if self._v_channels is None:
            self._v_channels = self._qk_channels
        # Project the output of QKV attention to a desired number of channels.
        # Default to the same number as the output of the QKV attention operation.
        if self._output_channels is None:
            self._output_channels = self._v_channels

        if self._qk_channels % self._num_heads != 0:
            raise ValueError(
                f"qk_channels ({self._qk_channels}) must be divisible by"
                f" num_heads ({self._num_heads})."
            )
        if self._v_channels % self._num_heads != 0:
            raise ValueError(
                f"v_channels ({self._v_channels}) must be divisible by"
                f" num_heads ({self._num_heads})."
            )
        qk_channels_per_head = self._qk_channels // self._num_heads
        v_channels_per_head = self._v_channels // self._num_heads

        # Project QKV to a common feature dimension.
        q = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_q)
        k = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_kv)
        v = conv_1d(self._v_channels, init_scale=self._init_scale)(inputs_kv)

        # Reshape channels for multi-head attention.
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape
        q = jnp.reshape(q, [batch, q_time, self._num_heads, qk_channels_per_head])
        k = jnp.reshape(k, [batch, kv_time, self._num_heads, qk_channels_per_head])
        v = jnp.reshape(v, [batch, kv_time, self._num_heads, v_channels_per_head])

        result = attend(
            q, k, v, dropout_prob=self._dropout_prob, attention_mask=attention_mask
        )
        return conv_1d(
            self._output_channels,
            with_bias=self._with_final_bias,
            init_scale=self._final_init_scale,
        )(result)


class MLP(hk.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, widening_factor=4, dropout_prob=0.0, init_scale=1.0, name=None):
        super(MLP, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._init_scale = init_scale

    def __call__(self, x, *, is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        output_channels = x.shape[-1]
        x = conv_1d(
            output_channels=self._widening_factor * output_channels,
            init_scale=self._init_scale,
        )(x)
        x = jax.nn.gelu(x)
        x = conv_1d(output_channels=output_channels, init_scale=self._init_scale)(x)
        return hk.dropout(hk.next_rng_key(), dropout_prob, x)


class SelfAttention(hk.Module):
    """A self-attention module, including a dense block."""

    def __init__(
        self,
        widening_factor=4,
        dropout_prob=0.0,
        dropout_attn_prob=0.0,
        num_heads=8,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        qk_channels=None,
        v_channels=None,
        name=None,
    ):
        super(SelfAttention, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._qk_channels = qk_channels
        self._v_channels = v_channels

    def __call__(self, inputs, *, attention_mask=None, is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

        x = inputs
        qkv_inputs = layer_norm(inputs)
        attention = Attention(
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            dropout_prob=dropout_attn_prob,
        )(qkv_inputs, qkv_inputs, attention_mask=attention_mask)
        attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)
        x += attention

        x += MLP(
            widening_factor=self._widening_factor,
            dropout_prob=dropout_prob,
            init_scale=self._dense_init_scale,
        )(layer_norm(x), is_training=is_training)
        return x


class CrossAttention(hk.Module):
    """A cross-attention module, including a dense block."""

    def __init__(
        self,
        widening_factor=1,
        dropout_prob=0.0,
        dropout_attn_prob=0.0,
        num_heads=8,
        att_init_scale=1.0,
        dense_init_scale=1.0,
        shape_for_attn="kv",
        use_query_residual=True,
        qk_channels=None,
        v_channels=None,
        name=None,
    ):
        super(CrossAttention, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._shape_for_attn = shape_for_attn
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels

    def __call__(self, inputs_q, inputs_kv, *, attention_mask=None, is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

        output_channels = inputs_q.shape[-1]
        if self._shape_for_attn == "q":
            qk_channels = inputs_q.shape[-1]
        elif self._shape_for_attn == "kv":
            qk_channels = inputs_kv.shape[-1]
        else:
            raise ValueError(
                f"Unknown value {self._shape_for_attn} for " "shape_for_attention."
            )

        v_channels = None
        if self._qk_channels is not None:
            qk_channels = self._qk_channels
        if self._v_channels is not None:
            v_channels = self._v_channels

        attention = Attention(
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            dropout_prob=dropout_attn_prob,
            qk_channels=qk_channels,
            v_channels=v_channels,
            output_channels=output_channels,
        )(layer_norm(inputs_q), layer_norm(inputs_kv), attention_mask=attention_mask)
        attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self._use_query_residual:
            x = inputs_q + attention
        else:
            x = attention

        x += MLP(
            widening_factor=self._widening_factor,
            dropout_prob=dropout_prob,
            init_scale=self._dense_init_scale,
        )(layer_norm(x), is_training=is_training)
        return x


#  -----------------------------------------------------------
#  -----------------------  Perceiver  -----------------------
#  -----------------------------------------------------------


class Perceiver(hk.Module):
    """The Perceiver: a scalable, fully attentional architecture."""

    def __init__(
        self,
        encoder,
        decoder,
        input_preprocessor=None,
        output_postprocessor=None,
        name="perceiver",
    ):
        super().__init__(name=name)

        # Feature and task parameters:
        self._input_preprocessor = input_preprocessor
        self._output_postprocessor = output_postprocessor
        self._decoder = decoder
        self._encoder = encoder

    def __call__(
        self,
        inputs,
        *,
        is_training,
        subsampled_output_points=None,
        pos=None,
        input_mask=None,
        query_mask=None,
    ):
        if self._input_preprocessor:
            network_input_is_1d = self._encoder._input_is_1d
            inputs, modality_sizes, inputs_without_pos = self._input_preprocessor(
                inputs,
                pos=pos,
                is_training=is_training,
                network_input_is_1d=network_input_is_1d,
            )
        else:
            modality_sizes = None
            inputs_without_pos = None

        # Get the queries for encoder and decoder cross-attends.
        encoder_query = self._encoder.latents(inputs)
        decoder_query = self._decoder.decoder_query(
            inputs,
            modality_sizes,
            inputs_without_pos,
            subsampled_points=subsampled_output_points,
        )

        # Run the network forward:
        z = self._encoder(
            inputs, encoder_query, is_training=is_training, input_mask=input_mask
        )
        _, output_modality_sizes = self._decoder.output_shape(inputs)
        output_modality_sizes = output_modality_sizes or modality_sizes

        outputs = self._decoder(
            decoder_query, z, is_training=is_training, query_mask=query_mask
        )

        if self._output_postprocessor:
            outputs = self._output_postprocessor(
                outputs, is_training=is_training, modality_sizes=output_modality_sizes
            )

        return outputs


class PerceiverEncoder(hk.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        # The encoder has a total of
        #   num_self_attends_per_block * num_blocks
        # self-attend layers. We share weights between blocks.
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn="kv",
        use_query_residual=True,
        name="perceiver_encoder",
    ):
        super().__init__(name=name)

        # Check that we can use multihead-attention with these shapes.
        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(
                f"num_z_channels ({num_z_channels}) must be divisible by"
                f" num_self_attend_heads ({num_self_attend_heads})."
            )
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(
                f"num_z_channels ({num_z_channels}) must be divisible by"
                f" num_cross_attend_heads ({num_cross_attend_heads})."
            )

        self._input_is_1d = True

        self._num_blocks = num_blocks

        # Construct the latent array initial state.
        self.z_pos_enc = TrainablePositionEncoding(
            index_dim=z_index_dim,
            num_channels=num_z_channels,
            init_scale=z_pos_enc_init_scale,
        )

        # Construct the cross attend:
        self.cross_attend = CrossAttention(
            dropout_prob=dropout_prob,
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            shape_for_attn=cross_attention_shape_for_attn,
            qk_channels=qk_channels,
            v_channels=v_channels,
            use_query_residual=use_query_residual,
        )

        # Construct the block of self-attend layers.
        # We get deeper architectures by applying this block more than once.
        self.self_attends = []
        for _ in range(num_self_attends_per_block):
            self_attend = SelfAttention(
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                qk_channels=qk_channels,
                v_channels=v_channels,
                widening_factor=self_attend_widening_factor,
            )
            self.self_attends.append(self_attend)

    def latents(self, inputs):
        # Initialize the latent array for the initial cross-attend.
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def __call__(self, inputs, z, *, is_training, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=jnp.ones(z.shape[:2], dtype=jnp.int32), kv_mask=input_mask
            )
        z = self.cross_attend(
            z, inputs, is_training=is_training, attention_mask=attention_mask
        )
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z, is_training=is_training)
        return z


class AbstractPerceiverDecoder(hk.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, query, z, *, is_training, query_mask=None):
        raise NotImplementedError


class ProjectionDecoder(AbstractPerceiverDecoder):
    """Baseline projection decoder (no cross-attention)."""

    def __init__(
        self, num_classes, final_avg_before_project=False, name="projection_decoder"
    ):
        super().__init__(name=name)
        self._final_avg_before_project = final_avg_before_project
        self._num_classes = num_classes
        self.final_layer = hk.Linear(num_classes, w_init=jnp.zeros, name="logits")

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        return None

    def output_shape(self, inputs):
        return ((inputs.shape[0], self._num_classes), None)

    def __call__(self, query, z, *, is_training, query_mask=None):
        # b x n_z x c -> b x c
        z = jnp.mean(z, axis=1, dtype=z.dtype)
        # b x c -> b x n_logits
        logits = self.final_layer(z)
        return logits


class BasicDecoder(AbstractPerceiverDecoder):
    """Cross-attention-based decoder."""

    def __init__(
        self,
        output_num_channels,
        position_encoding_type="trainable",
        # Ignored if position_encoding_type == 'none':
        output_index_dims=None,
        subsampled_index_dims=None,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        use_query_residual=False,
        output_w_init=None,
        concat_preprocessed_input=False,
        num_heads=1,
        name="basic_decoder",
        final_project=True,
        **position_encoding_kwargs,
    ):
        super().__init__(name=name)
        self._position_encoding_type = position_encoding_type

        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_pos_enc = None
        if self._position_encoding_type != "none":
            self.output_pos_enc = build_position_encoding(
                position_encoding_type,
                index_dims=output_index_dims,
                **position_encoding_kwargs,
            )

        self._output_index_dim = output_index_dims
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self._subsampled_index_dims = subsampled_index_dims
        self._output_num_channels = output_num_channels
        self._output_w_init = output_w_init
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._final_project = final_project
        self._num_heads = num_heads

        self._concat_preprocessed_input = concat_preprocessed_input

    def output_shape(self, inputs):
        return (
            (inputs[0], self._subsampled_index_dims, self._output_num_channels),
            None,
        )

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        assert self._position_encoding_type != "none"  # Queries come from elsewhere
        if subsampled_points is not None:
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            pos = jnp.stack(
                jnp.unravel_index(subsampled_points, self._output_index_dim), axis=1
            )
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / jnp.array(self._output_index_dim)[None, :]
            pos = jnp.broadcast_to(
                pos[None], [inputs.shape[0], pos.shape[0], pos.shape[1]]
            )
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0], pos=pos)
            pos_emb = jnp.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0])
        if self._concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError(
                    "Value is required for inputs_without_pos if"
                    " concat_preprocessed_input is True"
                )
            pos_emb = jnp.concatenate([inputs_without_pos, pos_emb], axis=-1)

        return pos_emb

    def __call__(self, query, z, *, is_training, query_mask=None):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        # Construct cross attention and linear layer lazily, in case we don't need
        # them.
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=query_mask, kv_mask=jnp.ones(z.shape[:2], dtype=jnp.int32)
            )
        decoding_cross_attn = CrossAttention(
            dropout_prob=0.0,
            num_heads=self._num_heads,
            widening_factor=1,
            shape_for_attn="kv",
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            use_query_residual=self._use_query_residual,
        )
        final_layer = hk.Linear(
            self._output_num_channels, w_init=self._output_w_init, name="output"
        )
        output = decoding_cross_attn(
            query, z, is_training=is_training, attention_mask=attention_mask
        )
        if self._final_project:
            output = final_layer(output)
        return output


class ClassificationDecoder(AbstractPerceiverDecoder):
    """Cross-attention based classification decoder.

    Light-weight wrapper of `BasicDecoder` for logit output.
    """

    def __init__(self, num_classes, name="classification_decoder", **decoder_kwargs):
        super().__init__(name=name)

        self._num_classes = num_classes
        self.decoder = BasicDecoder(
            output_index_dims=(1,),  # Predict a single logit array.
            output_num_channels=num_classes,
            **decoder_kwargs,
        )

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        return self.decoder.decoder_query(
            inputs,
            modality_sizes,
            inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    def output_shape(self, inputs):
        return (inputs.shape[0], self._num_classes), None

    def __call__(self, query, z, *, is_training, query_mask=None):
        # B x 1 x num_classes -> B x num_classes
        logits = self.decoder(query, z, is_training=is_training)
        return logits[:, 0, :]


class MultimodalDecoder(AbstractPerceiverDecoder):
    """Multimodal decoding by composing uni-modal decoders.

    The modalities argument of the constructor is a dictionary mapping modality
    name to the decoder of that modality. That decoder will be used to construct
    queries for that modality. However, there is a shared cross attention across
    all modalities, using the concatenated per-modality query vectors.
    """

    def __init__(
        self,
        modalities,
        num_outputs,
        output_num_channels,
        min_padding_size=2,
        subsampled_index_dims=None,
        name="multimodal_decoder",
        **decoder_kwargs,
    ):
        super().__init__(name=name)
        self._modalities = modalities
        self._subsampled_index_dims = subsampled_index_dims
        self._min_padding_size = min_padding_size
        self._output_num_channels = output_num_channels
        self._num_outputs = num_outputs
        self._decoder = BasicDecoder(
            output_index_dims=(num_outputs,),
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            **decoder_kwargs,
        )

    def decoder_query(
        self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None
    ):
        # Partition the flat inputs among the different modalities
        inputs = restructure(modality_sizes, inputs)
        # Obtain modality-specific decoders' queries
        subsampled_points = subsampled_points or dict()
        decoder_queries = dict()
        for modality, decoder in self._modalities.items():
            # Get input_without_pos for this modality if it exists.
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            decoder_queries[modality] = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,
                inputs_without_pos=input_without_pos,
                subsampled_points=subsampled_points.get(modality, None),
            )

        # Pad all queries with trainable position encodings to make them
        # have the same channels
        num_channels = (
            max(query.shape[2] for query in decoder_queries.values())
            + self._min_padding_size
        )

        def embed(modality, x):
            x = jnp.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = TrainablePositionEncoding(
                1,
                num_channels=num_channels - x.shape[2],
                init_scale=0.02,
                name=f"{modality}_padding",
            )(x.shape[0])
            pos = jnp.broadcast_to(
                pos, [x.shape[0], x.shape[1], num_channels - x.shape[2]]
            )
            return jnp.concatenate([x, pos], axis=2)

        # Apply a predictable ordering to the modalities
        return jnp.concatenate(
            [
                embed(modality, decoder_queries[modality])
                for modality in sorted(self._modalities.keys())
            ],
            axis=1,
        )

    def output_shape(self, inputs):
        if self._subsampled_index_dims is not None:
            subsampled_index_dims = sum(self._subsampled_index_dims.values())
        else:
            subsampled_index_dims = self._num_outputs
        return (
            (inputs.shape[0], subsampled_index_dims, self._output_num_channels),
            self._subsampled_index_dims,
        )

    def __call__(self, query, z, *, is_training, query_mask=None):
        # B x 1 x num_classes -> B x num_classes
        return self._decoder(query, z, is_training=is_training)


class BasicVideoAutoencodingDecoder(AbstractPerceiverDecoder):
    """Cross-attention based video-autoencoding decoder.

    Light-weight wrapper of `BasicDecoder` with video reshaping logic.
    """

    def __init__(
        self,
        output_shape,
        position_encoding_type,
        name="basic_video_autoencoding_decoder",
        **decoder_kwargs,
    ):
        super().__init__(name=name)
        if len(output_shape) != 4:  # B, T, H, W
            raise ValueError(f"Expected rank 4 output_shape, got {output_shape}.")
        # Build the decoder components:
        self._output_shape = output_shape
        self._output_num_channels = decoder_kwargs["output_num_channels"]

        self.decoder = BasicDecoder(
            output_index_dims=self._output_shape[1:4],  # T*H*W
            position_encoding_type=position_encoding_type,
            **decoder_kwargs,
        )

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        return self.decoder.decoder_query(
            inputs,
            modality_sizes=modality_sizes,
            inputs_without_pos=inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    def output_shape(self, inputs):
        return (
            [inputs.shape[0]] + self._output_shape[1:] + [self._output_num_channels],
            None,
        )

    def __call__(self, query, z, *, is_training, query_mask=None):
        output = self.decoder(query, z, is_training=is_training)

        output = jnp.reshape(output, self._output_shape + [output.shape[-1]])
        return output


class FlowDecoder(AbstractPerceiverDecoder):
    """Cross-attention based flow decoder."""

    def __init__(
        self,
        output_image_shape,
        output_num_channels=2,
        rescale_factor=100.0,
        name="flow_decoder",
        **decoder_kwargs,
    ):
        super().__init__(name=name)

        self._output_image_shape = output_image_shape
        self._output_num_channels = output_num_channels
        self._rescale_factor = rescale_factor
        self.decoder = BasicDecoder(
            output_num_channels=output_num_channels, **decoder_kwargs
        )

    def output_shape(self, inputs):
        # The channel dimensions of output here don't necessarily correspond to
        # (u, v) of flow: they may contain dims needed for the post-processor.
        return (
            (inputs.shape[0],)
            + tuple(self._output_image_shape)
            + (self._output_num_channels,),
            None,
        )

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")
        # assumes merged in time
        return inputs

    def __call__(self, query, z, *, is_training, query_mask=None):
        # Output flow and rescale.
        preds = self.decoder(query, z, is_training=is_training)
        preds /= self._rescale_factor

        return preds.reshape(
            [preds.shape[0]] + list(self._output_image_shape) + [preds.shape[-1]]
        )


###############################################################################
#
# Demo model definition
#
###############################################################################


class PerceiverBackbone(hk.Module):
    """Perceiver image preprocessor + encoder"""

    def __init__(
        self,
        encoder,
        input_preprocessor,
        name="perceiver",
    ):
        super().__init__(name=name)

        # Feature parameters:
        self._input_preprocessor = input_preprocessor
        self._encoder = encoder

    def __call__(
        self,
        inputs,
        *,
        is_training,
        pos=None,
        input_mask=None,
    ):
        network_input_is_1d = self._encoder._input_is_1d
        inputs, _, _ = self._input_preprocessor(
            inputs,
            pos=pos,
            is_training=is_training,
            network_input_is_1d=network_input_is_1d,
        )

        # Get the queries for encoder and decoder cross-attends.
        encoder_query = self._encoder.latents(inputs)

        # Run the network forward:
        z = self._encoder(
            inputs, encoder_query, is_training=is_training, input_mask=input_mask
        )

        return z


###############################################################################
#
# Demo model configuration
#
###############################################################################

fourier_pos_configs = dict(
    input_preprocessor=dict(
        position_encoding_type="fourier",
        fourier_position_encoding_kwargs=dict(
            concat_pos=True, max_resolution=(224, 224), num_bands=64, sine_only=False
        ),
        prep_type="pixels",
        spatial_downsample=1,
    ),
    encoder=dict(
        cross_attend_widening_factor=1,
        cross_attention_shape_for_attn="kv",
        dropout_prob=0,
        num_blocks=8,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        num_self_attends_per_block=6,
        num_z_channels=1024,
        self_attend_widening_factor=1,
        use_query_residual=True,
        z_index_dim=512,
        z_pos_enc_init_scale=0.02,
    ),
)


def perceiver_backbone(images):
    config = fourier_pos_configs
    input_preprocessor = ImagePreprocessor(**config["input_preprocessor"])
    encoder = PerceiverEncoder(**config["encoder"])
    model = PerceiverBackbone(encoder=encoder, input_preprocessor=input_preprocessor)
    logits = model(images, is_training=False)
    return logits


perceiver_backbone = hk.transform(perceiver_backbone)
key = jax.random.PRNGKey(42)
