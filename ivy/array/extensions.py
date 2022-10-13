# global
import abc
from typing import Optional, Union, Tuple, Iterable, Callable, Literal, Sequence
from numbers import Number

# local
import ivy


class ArrayWithExtensions(abc.ABC):
    def sinc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the sinc of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
        >>> y = x.sinc()
        >>> print(y)
        ivy.array([0.637,-0.212,0.127,-0.0909])
        """
        return ivy.sinc(self._data, out=out)

    def flatten(
        self: ivy.Array,
        *,
        start_dim: int,
        end_dim: int,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.flatten. This method simply
        wraps the function, and so the docstring for ivy.unstack also applies to
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

    def lcm(
        self: ivy.Array, x2: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array.
        x2
            second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            an array that includes the element-wise least common multiples
            of 'self' and x2

        Examples
        --------
        >>> x1=ivy.array([2, 3, 4])
        >>> x2=ivy.array([5, 8, 15])
        >>> x1.lcm(x2)
        ivy.array([10, 21, 60])
        """
        return ivy.lcm(self, x2, out=out)

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
        
    def pad(
        self: ivy.Array,
        /,
        pad_width: Union[Iterable[Tuple[int]], int],
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
        )

    def moveaxis(
        self: ivy.Array,
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.moveaxis. This method simply
        wraps the function, and so the docstring for ivy.unstack also applies to
        this method with minimal changes.

        Parameters
        ----------
        a
            The array whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with moved axes. This array is a view of the input array.

        Examples
        --------
        >>> x = ivy.zeros((3, 4, 5))
        >>> x.moveaxis(0, -1).shape
        (4, 5, 3)
        >>> x.moveaxis(-1, 0).shape
        (5, 3, 4)
        """
        return ivy.moveaxis(self._data, source, destination, out=out)
