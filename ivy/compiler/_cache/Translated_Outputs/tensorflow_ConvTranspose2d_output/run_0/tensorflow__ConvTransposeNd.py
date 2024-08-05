import tensorflow


from .tensorflow__ConvNd import tensorflow__ConvNd
from .tensorflow__helpers import tensorflow__ntuple_parse
from .tensorflow__helpers import tensorflow_dim_frnt_
from .tensorflow__helpers import tensorflow_get_item
from .tensorflow__helpers import tensorflow_size_frnt_
from .tensorflow__helpers import tensorflow_store_config_info

_single = tensorflow__ntuple_parse(1, "_single")


class tensorflow__ConvTransposeNd(tensorflow__ConvNd):
    @tensorflow_store_config_info
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ):
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _output_padding(
        self,
        input,
        output_size,
        stride,
        padding,
        kernel_size,
        num_spatial_dims,
        dilation=None,
    ):
        if output_size is None:
            ret = _single(self.output_padding)
        else:
            with tensorflow.name_scope("has_batch_dim"):
                has_batch_dim = tensorflow_dim_frnt_(input) == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                with tensorflow.name_scope("output_size"):
                    output_size = tensorflow_get_item(
                        output_size, slice(num_non_spatial_dims, None, None)
                    )
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {tensorflow_dim_frnt_(input)}D input, output_size must have {num_spatial_dims} or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )
            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                with tensorflow.name_scope("dim_size"):
                    dim_size = (
                        (tensorflow_size_frnt_(input, d + num_non_spatial_dims) - 1)
                        * tensorflow_get_item(stride, d)
                        - 2 * tensorflow_get_item(padding, d)
                        + (
                            tensorflow_get_item(dilation, d)
                            if dilation is not None
                            else 1
                        )
                        * (tensorflow_get_item(kernel_size, d) - 1)
                        + 1
                    )
                min_sizes.append(dim_size)
                max_sizes.append(
                    tensorflow_get_item(min_sizes, d)
                    + tensorflow_get_item(stride, d)
                    - 1
                )
            for i in range(len(output_size)):
                with tensorflow.name_scope("size"):
                    size = tensorflow_get_item(output_size, i)
                with tensorflow.name_scope("min_size"):
                    min_size = tensorflow_get_item(min_sizes, i)
                with tensorflow.name_scope("max_size"):
                    max_size = tensorflow_get_item(max_sizes, i)
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range from {min_sizes} to {max_sizes} (for an input of {tensorflow_size_frnt_(input)[2:]})"
                    )
            res = []
            for d in range(num_spatial_dims):
                res.append(
                    tensorflow_get_item(output_size, d)
                    - tensorflow_get_item(min_sizes, d)
                )
            ret = res
        return ret
