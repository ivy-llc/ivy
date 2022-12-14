# global
import abc
from typing import Optional, Union, Tuple, Literal

# local
import ivy


class ArrayWithLayersExperimental(abc.ABC):
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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of `ivy.max_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool2d` also applies
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
        >>> x = ivy.arange(12).reshape((2, 1, 3, 2))
        >>> print(x.max_pool2d((2, 2), (1, 1), 'SAME'))
        ivy.array([[[[ 2,  3],
                 [ 4,  5],
                 [ 4,  5]]],


               [[[ 8,  9],
                 [10, 11],
                 [10, 11]]]])

        >>> x = ivy.arange(48).reshape((2, 4, 3, 2))
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
        >>> x = ivy.arange(12).reshape((2, 1, 3, 2))
        >>> print(x.max_pool2d((2, 2), (1, 1), 'SAME'))
        ivy.array([[[[ 2,  3],
        [ 4,  5],
        [ 4,  5]]],
        [[[ 8,  9],
        [10, 11],
        [10, 11]]]])

        >>> x = ivy.arange(48).reshape((2, 4, 3, 2))
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
        type: Optional[Literal[1, 2, 3, 4]] = 2,
        n: Optional[int] = None,
        axis: Optional[int] = -1,
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
