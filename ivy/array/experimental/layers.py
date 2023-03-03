# global
import abc
from typing import Optional, Union, Tuple, Literal, Sequence

# local
import ivy


class _ArrayWithLayersExperimental(abc.ABC):
    def max_pool1d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.max_pool1d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool1d` also applies
        to this method with minimal changes.

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
            out=out,
        )

    def max_pool2d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.max_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool2d` also applies
        to this method with minimal changes.

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
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Computes a 3-D max pool given 5-D input x.

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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.avg_pool1d`. This method simply
        wraps the function, and so the docstring for `ivy.avg_pool1d` also applies
        to this method with minimal changes.

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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.avg_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.avg_pool2d` also applies
        to this method with minimal changes.

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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Computes a 3-D max pool given 5-D input x.

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
            The lenght of the transform. If n is less than the input signal lenght,
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
        """
        Computes the discrete Fourier transform of input.

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
        mode: Union[Literal["linear", "bilinear", "trilinear", "nearest"]] = "linear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Down/up samples the input to the given size.
        The algorithm used for interpolation is determined by mode.

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
            align_corners=align_corners,
            antialias=antialias,
            out=out,
        )

    def adaptive_avg_pool1d(
        self: ivy.Array,
        output_size: int,
    ) -> ivy.Array:
        """
        Applies a 1D adaptive average pooling over an input signal composed of several
        input planes.

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
    ) -> ivy.Array:
        """
        Applies a 2D adaptive average pooling over an input signal composed of several
        input planes.

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
        return ivy.adaptive_avg_pool2d(
            self._data,
            output_size,
        )
