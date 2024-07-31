from .ivy__ConvNd import ivy__ConvNd
from .ivy__helpers import ivy_dim_frnt_
from .ivy__helpers import ivy_parse
from .ivy__helpers import ivy_size_frnt_


class ivy__ConvTransposeNd(ivy__ConvNd):
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
            ret = ivy_parse(self.output_padding)
        else:
            has_batch_dim = ivy_dim_frnt_(input) == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {ivy_dim_frnt_(input)}D input, output_size must have {num_spatial_dims} or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )
            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                dim_size = (
                    (ivy_size_frnt_(input, d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)
            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range from {min_sizes} to {max_sizes} (for an input of {ivy_size_frnt_(input)[2:]})"
                    )
            res = []
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])
            ret = res
        return ret
