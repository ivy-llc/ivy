# global
import abc
from typing import Optional, Union, Tuple, List, Literal, Sequence, Callable

# local
import ivy


class _ArrayWithLayersExperimental(abc.ABC):
    def max_pool1d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
        /,
        *,
        data_format: str = "NWC",
        dilation: Union[int, Tuple[int]] = 1,
        ceil_mode: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of `ivy.max_pool1d`. This method
        simply wraps the function, and so the docstring for `ivy.max_pool1d`
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,w,d_in]*.
        kernel
            The size of the window for each dimension of the input tensor.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the max pooling operation.

        Examples
        --------
        >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
        >>> print(x.max_pool1d(2, 2, 'SAME'))
        ivy.array([[[ 4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11.]],
               [[16., 17., 18., 19.],
                [20., 21., 22., 23.]]])
        >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
        >>> print(x.max_pool1d(2, 2, 'VALID'))
        ivy.array([[[ 4.,  5.,  6.,  7.]],
           [[16., 17., 18., 19.]]])
        """
        return ivy.max_pool1d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            dilation=dilation,
            ceil_mode=ceil_mode,
            out=out,
        )

    def max_pool2d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
        /,
        *,
        data_format: str = "NHWC",
        dilation: Union[int, Tuple[int, ...]] = 1,
        ceil_mode: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of `ivy.max_pool2d`. This method
        simply wraps the function, and so the docstring for `ivy.max_pool2d`
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d_in]*.
        kernel
            The size of the window for each dimension of the input tensor.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the max pooling operation.

        Examples
        --------
        >>> x = ivy.arange(12.).reshape((2, 1, 3, 2))
        >>> print(x.max_pool2d((2, 2), (1, 1), 'SAME'))
        ivy.array([[[[ 2.,  3.],
                 [ 4.,  5.],
                 [ 4.,  5.]]],


               [[[ 8.,  9.],
                 [10., 11.],
                 [10., 11.]]]])

        >>> x = ivy.arange(48.).reshape((2, 4, 3, 2))
        >>> print(x.max_pool2d(3, 1, 'VALID'))
        ivy.array([[[[16., 17.]],

                [[22., 23.]]],


               [[[40., 41.]],

                [[46., 47.]]]])
        """
        return ivy.max_pool2d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            dilation=dilation,
            ceil_mode=ceil_mode,
            out=out,
        )

    def max_pool3d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
        /,
        *,
        data_format: str = "NDHWC",
        dilation: Union[int, Tuple[int, ...]] = 1,
        ceil_mode: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute a 3-D max pool given 5-D input x.

        Parameters
        ----------
        self
            Input volume *[batch_size,d,h,w,d_in]*.
        kernel
            Convolution filters *[d,h,w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> x = ivy.arange(48.).reshape((2, 3, 2, 2, 2))
        >>> print(x.max_pool3d(2, 2, 'VALID'))
        ivy.array([[[[[14., 15.]]]],
           [[[[38., 39.]]]]])
        >>> print(x.max_pool3d(2, 2, 'SAME'))
        ivy.array([[[[[14., 15.]]],
            [[[22., 23.]]]],
           [[[[38., 39.]]],
            [[[46., 47.]]]]])
        """
        return ivy.max_pool3d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            dilation=dilation,
            ceil_mode=ceil_mode,
            out=out,
        )

    def avg_pool1d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of `ivy.avg_pool1d`. This method
        simply wraps the function, and so the docstring for `ivy.avg_pool1d`
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,w,d_in]*.
        kernel
            The size of the window for each dimension of the input tensor.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the max pooling operation.

        Examples
        --------
        >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
        >>> print(x.avg_pool1d(2, 2, 'SAME'))
        ivy.array([[[ 2.,  3.,  4.,  5.],
                [ 8.,  9., 10., 11.]],
               [[14., 15., 16., 17.],
                [20., 21., 22., 23.]]])

        >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
        >>> print(x.avg_pool1d(2, 2, 'VALID'))
        ivy.array([[[ 2.,  3.,  4.,  5.]],
               [[14., 15., 16., 17.]]])
        """
        return ivy.avg_pool1d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            out=out,
        )

    def avg_pool2d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        divisor_override: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of `ivy.avg_pool2d`. This method
        simply wraps the function, and so the docstring for `ivy.avg_pool2d`
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d_in]*.
        kernel
            The size of the window for each dimension of the input tensor.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If given, it will be used as the divisor,
            otherwise kernel_size will be used.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the max pooling operation.

        Examples
        --------
        >>> x = ivy.arange(12.).reshape((2, 1, 3, 2))
        >>> print(x.max_pool2d((2, 2), (1, 1), 'SAME'))
        ivy.array([[[[ 2,  3],
        [ 4,  5],
        [ 4,  5]]],
        [[[ 8,  9],
        [10, 11],
        [10, 11]]]])

        >>> x = ivy.arange(48.).reshape((2, 4, 3, 2))
        >>> print(x.max_pool2d(3, 1, 'VALID'))
        ivy.array([[[[16, 17]],
        [[22, 23]]],
        [[[40, 41]],
        [[46, 47]]]])
        """
        return ivy.avg_pool2d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            out=out,
        )

    def avg_pool3d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        divisor_override: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute a 3-D max pool given 5-D input x.

        Parameters
        ----------
        self
            Input volume *[batch_size,d,h,w,d_in]*.
        kernel
            Convolution filters *[d,h,w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If specified, it will be used as divisor,
            otherwise kernel_size will be used.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> x = ivy.arange(48.).reshape((2, 3, 2, 2, 2))
        >>> print(x.avg_pool3d(2, 2, 'VALID'))
        ivy.array([[[[[ 7.,  8.]]]],
               [[[[31., 32.]]]]])
        >>> print(x.avg_pool3d(2, 2, 'SAME'))
        ivy.array([[[[[ 7.,  8.]]],
                [[[19., 20.]]]],
               [[[[31., 32.]]],
                [[[43., 44.]]]]])
        """
        return ivy.avg_pool3d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            out=out,
        )

    def dct(
        self: ivy.Array,
        /,
        *,
        type: Literal[1, 2, 3, 4] = 2,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Optional[Literal["ortho"]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.dct. This method simply
        wraps the function, and so the docstring for ivy.dct also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The input signal.
        type
            The type of the dct. Must be 1, 2, 3 or 4.
        n
            The length of the transform. If n is less than the input signal length,
            then x is truncated, if n is larger than x is zero-padded.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array containing the transformed input.

        Examples
        --------
        >>> x = ivy.array([8., 16., 24., 32., 40., 48., 56., 64.])
        >>> x.dct(type=2, norm="ortho")
        ivy.array([ 102.,  -51.5,   0.,  -5.39,   0.,  -1.61,   0., -0.406])
        """
        return ivy.dct(
            self._data,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            out=out,
        )

    def idct(
        self: ivy.Array,
        /,
        *,
        type: Literal[1, 2, 3, 4] = 2,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Optional[Literal["ortho"]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.idct. This method simply
        wraps the function, and so the docstring for ivy.idct also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The input signal.
        type
            The type of the idct. Must be 1, 2, 3 or 4.
        n
            The length of the transform. If n is less than the input signal length,
            then x is truncated, if n is larger than x is zero-padded.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array containing the transformed input.

        Examples
        --------
        >>> x = ivy.array([8., 16., 24., 32., 40., 48., 56., 64.])
        >>> x.idct(type=2, norm="ortho")
        ivy.array([ 79.49862671, -70.37691498,  30.00390816, -23.58938599,
            13.92713165, -10.078475  ,   5.19664812,  -1.95411837])
        """
        return ivy.idct(
            self._data,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            out=out,
        )

    def fft(
        self: ivy.Array,
        dim: int,
        /,
        *,
        norm: str = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.ifft. This method simply
        wraps the function, and so the docstring for ivy.ifft also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input volume *[...,d_in,...]*,
            where d_in indicates the dimension that needs FFT.
        dim
            The dimension along which to take the one dimensional FFT.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        n
            Optional argument indicating the sequence length, if given, the input
            would be padded with zero or truncated to length n before performing FFT.
            Should be a integer greater than 1.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Array containing the transformed input.

        Examples
        --------
        >>> a = ivy.array((np.exp(2j * np.pi * np.arange(8) / 8)))
        >>> a.fft(0)
        ivy.array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
                    2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
                    9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
                    1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])
        """
        return ivy.fft(
            self._data,
            dim,
            norm=norm,
            n=n,
            out=out,
        )

    def ifft(
        self: ivy.Array,
        dim: int,
        *,
        norm: str = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.ifft. This method simply
        wraps the function, and so the docstring for ivy.ifft also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input volume *[...,d_in,...]*,
            where d_in indicates the dimension that needs IFFT.
        dim
            The dimension along which to take the one dimensional IFFT.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        n
            Optional argument indicating the sequence length, if given, the input
            would be padded with zero or truncated to length n before performing IFFT.
            Should be a integer greater than 1.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Array containing the transformed input.

        Examples
        --------
        >>> a = ivy.array((np.exp(2j * np.pi * np.arange(8) / 8)))
        >>> a.ifft(0)
        ivy.array([-4.30636606e-17+1.43029718e-18j,  0.00000000e+00+1.53080850e-17j,
                    1.43029718e-18+1.53080850e-17j,  0.00000000e+00+9.58689626e-18j,
                    1.24474906e-17+2.91858728e-17j,  0.00000000e+00+1.53080850e-17j,
                    2.91858728e-17+1.53080850e-17j,  1.00000000e+00-1.01435406e-16j])
        """
        return ivy.ifft(
            self._data,
            dim,
            norm=norm,
            n=n,
            out=out,
        )

    def embedding(
        self: ivy.Array,
        indices: ivy.Array,
        /,
        *,
        max_norm: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.embedding(self._data, indices, max_norm=max_norm, out=out)

    def dft(
        self,
        /,
        *,
        axis: int = 1,
        inverse: bool = False,
        onesided: bool = False,
        dft_length: Optional[Union[int, Tuple[int]]] = None,
        norm: str = "backward",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute the discrete Fourier transform of input.

        Parameters
        ----------
        self
            Input volume *[...,d_in,...]*,
            where d_in indicates the dimension that needs FFT.
        axis
            The axis on which to perform the DFT. By default this
            value is  set to 1, which corresponds to the first dimension
            after the batch index.
        inverse
            Whether to perform the inverse discrete fourier transform.
            By default this value is set to False.
        onesided
            If onesided is True, only values for w in [0, 1, 2, …, floor(n_fft/2) + 1]
            are returned because the real-to-complex Fourier transform satisfies the
            conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]*. Note if the
            input or window tensors are complex, then onesided output is not possible.
            Enabling onesided with real inputs performs a Real-valued fast Fourier
            transform (RFFT). When invoked with real or complex valued input, the
            default value is False. Values can be True or False.
        dft_length
            The length of the signal.If greater than the axis dimension,
            the signal will be zero-padded up to dft_length. If less than
            the axis dimension, only the first dft_length values will be
            used as the signal. It’s an optional value.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The Fourier Transform of the input vector.If onesided is False,
            the following shape is expected: [batch_idx][signal_dim1][signal_dim2]
            …[signal_dimN][2]. If axis=0 and onesided is True, the following shape
            is expected: [batch_idx][floor(signal_dim1/2)+1][signal_dim2]
            …[signal_dimN][2]. If axis=1 and onesided is True, the following
            shape is expected: [batch_idx][signal_dim1] [floor(signal_dim2/2)+1]
            …[signal_dimN][2]. If axis=N-1 and onesided is True, the following
            shape is expected: [batch_idx][signal_dim1][signal_dim2]…
            [floor(signal_dimN/2)+1][2]. The signal_dim at the specified axis
            is equal to the dft_length.
        """
        return ivy.dft(
            self._data,
            axis=axis,
            inverse=inverse,
            onesided=onesided,
            dft_length=dft_length,
            norm=norm,
            out=out,
        )

    def interpolate(
        self,
        size: Union[Sequence[int], int],
        /,
        *,
        mode: Union[
            Literal[
                "linear",
                "bilinear",
                "trilinear",
                "nearest",
                "area",
                "nearest_exact",
                "tf_area",
                "bicubic",
            ]
        ] = "linear",
        scale_factor: Optional[Union[Sequence[int], int]] = None,
        recompute_scale_factor: Optional[bool] = None,
        align_corners: Optional[bool] = None,
        antialias: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Down/up samples the input to the given size. The algorithm used for
        interpolation is determined by mode.

        Parameters
        ----------
        self
            Input array, Must have the shape
            [batch x channels x [optional depth] x [optional height] x width].
        size
            Output size.
        mode
            Interpolation mode. Can be one of the following:
            - linear
            - bilinear
            - trilinear
            - nearest
            - area
            - tf_area
            - bicubic
            - mitchellcubic
            - lanczos3
            - lanczos5
            - gaussian
        scale_factor
            Multiplier for spatial size that defines the output size
            (overwriting `size`).
        align_corners
            If True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at the corner pixels. If False, the corner
            pixels are not aligned, and the interpolation uses edge value padding for
            out-of-boundary values.
            only has an effect when mode is 'linear', 'bilinear',
            'bicubic' or 'trilinear'. Default: False
        antialias
            If True, antialiasing is applied when downsampling an image.
            Supported modes: 'bilinear', 'bicubic'.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
            resized array
        """
        return ivy.interpolate(
            self._data,
            size,
            mode=mode,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute_scale_factor,
            align_corners=align_corners,
            antialias=antialias,
            out=out,
        )

    def adaptive_avg_pool1d(
        self: ivy.Array,
        output_size: int,
    ) -> ivy.Array:
        """Apply a 1D adaptive average pooling over an input signal composed of
        several input planes.

        Parameters
        ----------
        self
            Input array. Must have shape (N, C, L_in) or (C, L_in) where N is
            the batch dimension, C is the feature dimension, and L_in is the spatial
            dimension.
        output_size
            Spatial output size.

        Returns
        -------
            The result of the pooling operation. Will have shape (N, C, L_out) or
            (C, L_out), where L_out = `output_size`
        """
        return ivy.adaptive_avg_pool1d(
            self._data,
            output_size,
        )

    def adaptive_avg_pool2d(
        self: ivy.Array,
        output_size: Union[Sequence[int], int],
        /,
        *,
        data_format: str = "NHWC",
    ) -> ivy.Array:
        """Apply a 2D adaptive average pooling over an input signal composed of
        several input planes.

        Parameters
        ----------
        self
            A 3D or 4D input array. Should have a floating-point data type.
        output_size
            Spatial output size.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".

        Returns
        -------
            The result of the pooling operation. Will have shape (N, C, S_0, S_1) or
            (C, S_0, S_1), where S = `output_size`
        """
        return ivy.adaptive_avg_pool2d(
            self._data,
            output_size,
            data_format=data_format,
        )

    def adaptive_max_pool2d(
        self: ivy.Array,
        output_size: Union[Sequence[int], int],
    ) -> ivy.Array:
        """Apply a 2D adaptive maximum pooling over an input signal composed of
        several input planes.

        Parameters
        ----------
        self
            Input array. Must have shape (N, C, H_in, W_in) or (C, H_in, W_in) where N
            is the batch dimension, C is the feature dimension, and H_in and W_in are
            the 2 spatial dimensions.
        output_size
            Spatial output size.

        Returns
        -------
            The result of the pooling operation. Will have shape (N, C, S_0, S_1) or
            (C, S_0, S_1), where S = `output_size`
        """
        return ivy.adaptive_max_pool2d(
            self._data,
            output_size,
        )

    def adaptive_max_pool3d(
        self: ivy.Array,
        output_size: Union[Sequence[int], int],
    ) -> ivy.Array:
        return ivy.adaptive_max_pool3d(
            self._data,
            output_size,
        )

    def reduce_window(
        self: ivy.Array,
        init_value: Union[int, float],
        computation: Callable,
        window_dimensions: Union[int, Sequence[int]],
        /,
        *,
        window_strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
        base_dilation: Union[int, Sequence[int]] = 1,
        window_dilation: Union[int, Sequence[int]] = 1,
    ) -> ivy.Array:
        """Apply a reduction function to all elements in each window of an
        array.

        Parameters
        ----------
        self
            An array representing the base area on which the window is going to slide
            over.
        init_value
            The starting value for the reduction.
        computation
            The reduction function to apply to elements in each window.
        window_dimensions
            A sequence containing the window dimensions.
        window_strides
            A sequence containing the window strides.
        padding
            Either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
            padding), or a sequence of n (low, high) integer pairs that give the padding
            to apply before and after each spatial dimension.
        base_dilation
            A sequence containing the base dilation values.
        window_dilation
            A sequence containing the window dilation values.

        Returns
        -------
        ret
            The result of the pooling-like operation.

        Examples
        --------
        >>> x = ivy.array([[1, 2, 3, 4],
        >>>                [5, 6, 7, 8],
        >>>                [9, 10, 11, 12]])
        >>> x.reduce_window(0, ivy.sum, (2, 2))
        ivy.array([[32.]])
        """
        return ivy.reduce_window(
            self._data,
            init_value,
            computation,
            window_dimensions,
            window_strides=window_strides,
            padding=padding,
            base_dilation=base_dilation,
            window_dilation=window_dilation,
        )

    def fft2(
        self: ivy.Array,
        *,
        s: Optional[Sequence[int]] = None,
        dim: Sequence[int] = (-2, -1),
        norm: str = "backward",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute the 2-dimensional discrete Fourier Transform.

        Parameters
        ----------
        x
            Input volume *[...,d_in,...]*,
            where d_in indicates the dimension that needs FFT2.
        s
            sequence of ints, optional
            Shape (length of each transformed axis) of the output (s[0] refers
            to axis 0, s[1] to axis 1, etc.). This corresponds to n for fft(x, n).
            Along each axis, if the given shape is smaller than that of the input,
            the input is cropped. If it is larger, the input is padded with zeros.
            If s is not given, the shape of the input along the axes specified by
            axes is used.
        dim
            Axes over which to compute the FFT2. If not given, the last two axes are
            used. A repeated index in axes means the transform over that axis is
            performed multiple times. A one-element sequence means that a
            one-dimensional FFT is performed.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the FFT2 operation.

        Examples
        --------
        >>> a = ivy.array([[0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2],
                        [3, 3, 3, 3, 3],
                        [4, 4, 4, 4, 4]])
        >>> ivy.fft2(a)
        array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
                0.  +0.j        ,   0.  +0.j        ],
            [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
                0.  +0.j        ,   0.  +0.j        ],
            [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
                0.  +0.j        ,   0.  +0.j        ],
            [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
                0.  +0.j        ,   0.  +0.j        ],
            [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
                0.  +0.j        ,   0.  +0.j        ]])
        """
        return ivy.fft2(self._data, s=s, dim=dim, norm=norm, out=out)

    def ifftn(
        self: ivy.Array,
        s: Optional[Union[int, Tuple[int, ...]]] = None,
        axes: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        norm: str = "backward",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute the N-dimensional inverse discrete Fourier Transform.

        Parameters
        ----------
        x
              Input array of complex numbers.

        s
            sequence of ints, optional
            Shape (length of transformed axis) of the output (`s[0]` refers to axis 0,
            `s[1]` to axis 1, etc.). If given shape is smaller than that of the input,
            the input is cropped. If larger, input is padded with zeros. If `s` is not
            given, shape of input along axes specified by axes is used.
        axes
            axes over which to compute the IFFT. If not given, last `len(s)` axes are
            used, or all axes if `s` is also not specified. Repeated indices in axes
            means inverse transform over that axis is performed multiple times.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The truncated or zero-padded input, transformed along the axes indicated
            by axes, or by a combination of s or x, as explained in the parameters
            section above.

        Examples
        --------
        >>> x = ivy.array([[0.24730653+0.90832391j, 0.49495562+0.9039565j,
        ...                 0.98193269+0.49560517j],
        ...                 [0.93280757+0.48075343j, 0.28526384+0.3351205j,
        ...                 0.2343787 +0.83528011j],
        ...                 [0.18791352+0.30690572j, 0.82115787+0.96195183j,
        ...                 0.44719226+0.72654048j]])
        >>> y = ivy.ifftn(x)
        >>> print(y)
        ivy.array([[ 0.51476765+0.66160417j, -0.04319742-0.05411636j,
                -0.015561  -0.04216015j],
               [ 0.06310689+0.05347854j, -0.13392983+0.16052352j,
                -0.08371392+0.17252843j],
               [-0.0031429 +0.05421245j, -0.10446617-0.17747098j,
                 0.05344324+0.07972424j]])
        >>> x = ivy.array([[0.24730653+0.90832391j, 0.49495562+0.9039565j,
        ...                 0.98193269+0.49560517j],
        ...                 [0.93280757+0.48075343j, 0.28526384+0.3351205j,
        ...                 0.2343787 +0.83528011j],
        ...                 [0.18791352+0.30690572j, 0.82115787+0.96195183j,
        ...                 0.44719226+0.72654048j]])
        >>> y = ivy.ifftn(x, s=[2, 1], axes=[0, 1], norm='ortho')
        >>> print(y)
        ivy.array([[ 0.8344667 +0.98222595j],
               [-0.48472244+0.30233797j]])
        """
        return ivy.ifftn(self._data, s=s, axes=axes, norm=norm, out=out)

    def rfft(
        self: ivy.Array,
        /,
        *,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] = "backward",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.rfft. This method simply
        wraps the function, and so the docstring for ivy.rfft also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Must have a real-valued floating-point data type.
        n
            length of the transformed axis of the input. If
            -   n is greater than the length of the input array, the input array
            is zero-padded to length n.
            -   n is less than the length of the input array, the input array is
            trimmed to length n.
            -   n is not provided, the length of the transformed axis of the
            output must equal the length of the input along the axis specified
            by axis. Default is ``None``.
        axis
            axis (dimension) over which to compute the Fourier transform.
            If not set, the last axis (dimension) is used. Default is ``-1``.
        norm
            normalization mode. Should be one of the following modes:
            -   'backward': no normalization.
            -   'ortho': normalize by 1/sqrt(n) (i.e., make the FFT orthonormal).
            -   'forward': normalize by 1/n.
            Default is ``backward``.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array transformed along the axis (dimension) indicated by axis.
            The returned array must have a complex-valued floating-point
            data type determined by Type Promotion Rules.

        Examples
        --------
        >>> x = ivy.array([0,1,2])
        >>> y = x.rfft()
        >>> print(y)
        ivy.array([ 3. +0.j       , -1.5+0.8660254j])
        """
        return ivy.rfft(self, n=n, axis=axis, norm=norm, out=out)

    def rfftn(
        self: ivy.Array,
        s: Optional[Sequence[int]] = None,
        axes: Optional[Sequence[int]] = None,
        *,
        norm: str = "backward",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute the n-dimensional discrete Fourier Transform.

        Parameters
        ----------
        self
            Input array.
        s
            Shape (length of each transformed axis) of the output.
        axes
            Axes over which to compute the RFFT. If not given, the last len(s) axes are
            used.
        norm
            Normalization mode: "backward", "ortho", or "forward".
        out
            Optional output array for writing the result.

        Returns
        -------
        ret
            The result of the RFFT operation.
        """
        return ivy.rfftn(self._data, s=s, axes=axes, norm=norm, out=out)

    def stft(
        self: ivy.Array,
        frame_length: int,
        frame_step: int,
        /,
        *,
        fft_length: Optional[int] = None,
        window_fn: Optional[Callable] = None,
        pad_end: Optional[bool] = False,
        name: Optional[str] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Compute the Short-time Fourier Transform of signals.

        Parameters
        ----------
        self
            Input Arrays.
        frame_length
           An integer scalar Tensor. The window length in samples.
        frame_step
            An integer scalar Tensor. The number of samples to step.
        fft_length
            An integer scalar Tensor. The size of the FFT to apply.
            If not provided, uses the smallest power of 2 enclosing frame_length.
        window_fn
            A callable that takes a window length and a dtype keyword
            argument and returns a [window_length] Tensor of samples in the
            provided datatype. If set to None, no windowing is used.
        pad_end
            Whether to pad the end of signals with zeros when the provided frame length
            and step produces a frame that lies partially past its end.
        name
            An optional name for the operation.
        out
            Optional output array for writing the result.

        Returns
        -------
        ret
            A [..., frames, fft_unique_bins] Tensor of
            complex64/complex128 STFT values where fft_unique_bins is
            fft_length // 2 + 1 (the unique components of the FFT).
        """
        return ivy.stft(
            self._data,
            frame_length,
            frame_step,
            fft_length=fft_length,
            window_fn=window_fn,
            pad_end=pad_end,
            name=name,
            out=out,
        )

    def sliding_window(
        self: ivy.Array,
        window_size: Union[int, Tuple[int, int], Tuple[int, int, int]],
        /,
        *,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    ) -> ivy.Array:
        """Slide a window of specified dimension over all elements of an array.

        Parameters
        ----------
        input
            An array representing the base area on which the window is going to slide
            over.
        window_size
            Size of the sliding window for each dimension of the input.
        stride
            The stride of the sliding window for each dimension of input
        padding
            Either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’
            (no padding), or a sequence of n (low, high) integer pairs that give the
            padding to apply before and after each spatial dimension.
        dilation
            The stride between elements within a sliding window, must be > 0.

        Returns
        -------
        ret
            The result of the sliding window operation.

        Examples
        --------
        >>> x = ivy.array([[1, 2, 3, 4],
        >>>                [5, 6, 7, 8],
        >>>                [9, 10, 11, 12]])
        >>> x.sliding_window((2, 2))
        ivy.array([[[ 1,  2,  5,  6],
                    [ 2,  3,  6,  7],
                    [ 3,  4,  7,  8]],

                    [[ 5,  6,  9, 10],
                    [ 6,  7, 10, 11],
                    [ 7,  8, 11, 12]]])
        """
        return ivy.sliding_window(
            self._data,
            window_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

    def max_unpool1d(
        self: ivy.Array,
        indices: ivy.Array,
        kernel_size: Union[Tuple[int], int],
        /,
        *,
        strides: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
        data_format: Optional[str] = "NCW",
    ) -> ivy.Array:
        """Compute a 1-D max unpooling given the 1-D pooled input x and its
        indices.

        Parameters
        ----------
        self
            Pooled input image *[batch_size, w, d_in]*.
        indices
            Indices obtained from the corresponding max pooling operation.
        kernel_size
            Size of the kernel i.e., the sliding window for each
            dimension of input. *[w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list
            indicating the per-dimension paddings.
        data_format
            NWC" or "NCW". Defaults to "NWC".

        Returns
        -------
        ret
            The result of the unpooling operation.
        """
        return ivy.max_unpool1d(
            self._data,
            indices,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
