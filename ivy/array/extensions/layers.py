# global
import abc
from typing import Optional, Union, Tuple, Iterable, Callable, Literal, Any
from numbers import Number

# local
import ivy


class ArrayWithLayersExtensions(abc.ABC):
    def flatten(
        self: ivy.Array,
        *,
        start_dim: Optional[int] = 0,
        end_dim: Optional[int] = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.flatten. This method simply
        wraps the function, and so the docstring for ivy.flatten also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array to flatten.
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            the flattened array over the specified dimensions.

        Examples
        --------
        >>> x = ivy.array([1,2], [3,4])
        >>> ivy.flatten(x)
        ivy.array([1, 2, 3, 4])

        >>> x = ivy.array(
            [[[[ 5,  5,  0,  6],
            [17, 15, 11, 16],
            [ 6,  3, 13, 12]],

            [[ 6, 18, 10,  4],
            [ 5,  1, 17,  3],
            [14, 14, 18,  6]]],


        [[[12,  0,  1, 13],
            [ 8,  7,  0,  3],
            [19, 12,  6, 17]],

            [[ 4, 15,  6, 15],
            [ 0,  5, 17,  9],
            [ 9,  3,  6, 19]]],


        [[[17, 13, 11, 16],
            [ 4, 18, 17,  4],
            [10, 10,  9,  1]],

            [[19, 17, 13, 10],
            [ 4, 19, 16, 17],
            [ 2, 12,  8, 14]]]]
            )
        >>> ivy.flatten(x, start_dim = 1, end_dim = 2)
        ivy.array(
            [[[ 5,  5,  0,  6],
            [17, 15, 11, 16],
            [ 6,  3, 13, 12],
            [ 6, 18, 10,  4],
            [ 5,  1, 17,  3],
            [14, 14, 18,  6]],

            [[12,  0,  1, 13],
            [ 8,  7,  0,  3],
            [19, 12,  6, 17],
            [ 4, 15,  6, 15],
            [ 0,  5, 17,  9],
            [ 9,  3,  6, 19]],

            [[17, 13, 11, 16],
            [ 4, 18, 17,  4],
            [10, 10,  9,  1],
            [19, 17, 13, 10],
            [ 4, 19, 16, 17],
            [ 2, 12,  8, 14]]]))
        """
        return ivy.flatten(self._data, start_dim=start_dim, end_dim=end_dim, out=out)

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

    def max_pool1d(
        self: ivy.Array,
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
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

    def pad(
        self: ivy.Array,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        out: Optional[ivy.Array] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies
        to this method with minimal changes.
        """
        return ivy.pad(
            self._data,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            out=out,
            **kwargs,
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
        return ivy.dct(self._data, type=type, n=n, axis=axis, norm=norm, out=out)
