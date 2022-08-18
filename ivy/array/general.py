# global
import abc
import numpy as np
from numbers import Number
from typing import Any, Iterable, Union, Optional, Dict, Callable, List, Tuple

# ToDo: implement all methods here as public instance methods

# local
import ivy


class ArrayWithGeneral(abc.ABC):
    def all_equal(self: ivy.Array, x2: Iterable[Any], equality_matrix: bool = False) -> Union[bool, ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.all_equal. This method simply wraps the
        function, and so the docstring for ivy.all_equal also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        x2
            input iterable to compare to ``self``
        equality_matrix
            Whether to return a matrix of equalities comparing each input with every
            other. Default is False.

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix array of booleans if
            equality_matrix=True is set.

        """
        return ivy.all_equal(self, x2, equality_matrix=equality_matrix)

    def unstack(self: ivy.Array, axis: int, keepdims: bool = False) -> ivy.Array:
        """ivy.Array instance method variant of ivy.unstack. This method simply
        wraps the function, and so the docstring for ivy.unstack also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input array to unstack.
        axis
            Axis for which to unpack the array.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is False.

        Returns
        -------
        ret
            List of arrays, unpacked along specified dimensions.

        """
        return ivy.unstack(self._data, axis, keepdims)

    def gather(
        self: ivy.Array,
        indices: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gather. This method simply wraps the
        function, and so the docstring for ivy.gather also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            array, the array from which to gather values.
        indices
            array, index array.
        axis
            optional int, the axis from which to gather from. Default is -1.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            New array with the values gathered at the specified indices along
            the specified axis.
        """
        return ivy.gather(self._data, indices, axis, out=out)

    def gather_nd(
        self: ivy.Array,
        indices: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.gather_nd. This method simply wraps the
        function, and so the docstring for ivy.gather_nd also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The array from which to gather values.
        indices
            Index array.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as
            ``x`` if None.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            New array of given shape, with the values gathered at the indices.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([1])
        >>> z = x.gather_nd(y)
        >>> print(z)
        ivy.array(2)
        """
        return ivy.gather_nd(self, indices, out=out)

    def einops_rearrange(
        self: ivy.Array,
        pattern: str,
        *,
        out: Optional[ivy.Array] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.einops_rearrange.
        This method simply wraps the function, and so the docstring
        for ivy.einops_rearrange also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input array to be re-arranged.
        pattern
            Rearrangement pattern.
        axes_lengths
            Any additional specifications for dimensions.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New array with einops.rearrange having been applied.

        """
        return ivy.einops_rearrange(self._data, pattern, out=out, **axes_lengths)

    def einops_reduce(
        self: ivy.Array,
        pattern: str,
        reduction: Union[str, Callable],
        *,
        out: Optional[ivy.Array] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.einops_reduce. This method simply
        wraps the function, and so the docstring for ivy.einops_reduce also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array to be reduced.
        pattern
            Reduction pattern.
        reduction
            One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or
            callable.
        axes_lengths
            Any additional specifications for dimensions.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New array with einops.reduce having been applied.

        """
        return ivy.einops_reduce(
            self._data, pattern, reduction, out=out, **axes_lengths
        )

    def einops_repeat(
        self: ivy.Array,
        pattern: str,
        *,
        out: Optional[ivy.Array] = None,
        **axes_lengths: Dict[str, int],
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.einops_repeat. This method simply
        wraps the function, and so the docstring for ivy.einops_repeat also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array to be repeated.
        pattern
            Rearrangement pattern.
        axes_lengths
            Any additional specifications for dimensions.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New array with einops.repeat having been applied.

        """
        return ivy.einops_repeat(self._data, pattern, out=out, **axes_lengths)

    def to_numpy(self: ivy.Array) -> np.ndarray:
        """
        ivy.Array instance method variant of ivy.to_numpy. This method simply wraps
        the function, and so the docstring for ivy.to_numpy also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.

        Returns
        -------
        ret
            a numpy array copying all the element of the array ``self``.

        """
        return ivy.to_numpy(self)

    def to_list(self: ivy.Array) -> List:
        """
        ivy.Array instance method variant of ivy.to_list. This method simply wraps
        the function, and so the docstring for ivy.to_list also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.

        Returns
        -------
        ret
            A list representation of the input array ``x``.
        """
        return ivy.to_list(self)

    def stable_divide(
        self,
        denominator: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container],
        min_denominator: Union[
            Number, ivy.Array, ivy.NativeArray, ivy.Container
        ] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.stable_divide. This method simply wraps
        the function, and so the docstring for ivy.stable_divide also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array, used as the numerator for division.
        denominator
            denominator for division.
        min_denominator
            the minimum denominator to use, use global ivy._MIN_DENOMINATOR by default.

        Returns
        -------
        ret
            a numpy array containing the elements of numerator divided by
            the corresponding element of denominator

        Examples
        --------
        >>> x = ivy.asarray([4., 5., 6.])
        >>> y = x.stable_divide(2)
        >>> print(y)
        ivy.array([2., 2.5, 3.])

        >>> x = ivy.asarray([4, 5, 6])
        >>> y = x.stable_divide(4, min_denominator=1)
        >>> print(y)
        ivy.array([0.8, 1. , 1.2])

        >>> x = ivy.asarray([[4., 5., 6.], [7., 8., 9.]])
        >>> y = ivy.asarray([[1., 2., 3.], [2., 3., 4.]])
        >>> z = x.stable_divide(y)
        >>> print(z)
        ivy.array([[4.  , 2.5 , 2.  ],
               [3.5 , 2.67, 2.25]])

        """
        return ivy.stable_divide(self, denominator, min_denominator=min_denominator)

    def clip_vector_norm(
        self: ivy.Array,
        max_norm: float,
        p: float = 2.0,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.clip_vector_norm. This method simply
        wraps the function, and so the docstring for ivy.clip_vector_norm also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array
        max_norm
            float, the maximum value of the array norm.
        p
            optional float, the p-value for computing the p-norm. Default is 2.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        Examples
        --------
        With :code:`ivy.Array` instance method:

        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.clip_vector_norm(2.0)
        >>> print(y)
        ivy.array([0.   , 0.894, 1.79 ])

        """
        return ivy.clip_vector_norm(self, max_norm, p, out=out)

    def array_equal(self: ivy.Array, x: Union[ivy.Array, ivy.NativeArray]) -> bool:
        """
        ivy.Array instance method variant of ivy.array_equal. This method simply wraps the
        function, and so the docstring for ivy.array_equal also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        x
            input array to compare to ``self``

        Returns
        -------
        ret
            Boolean, whether or not the input arrays are equal

        """
        return ivy.array_equal(self, x)

    def arrays_equal(self: ivy.Array, x: List[Union[ivy.Array, ivy.NativeArray]]) -> bool:
        """
        ivy.Array instance method variant of ivy.arrays_equal. This method simply wraps the
        function, and so the docstring for ivy.arrays_equal also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        x
            input list of arrays to compare to ``self``

        Returns
        -------
        ret
            Boolean, whether or not the input arrays are equal

        """
        return ivy.arrays_equal(List[self] + x)

    def assert_supports_inplace(self: ivy.Array) -> bool:
        """
        ivy.Array instance method variant of ivy.assert_supports_inplace. This method simply wraps the
        function, and so the docstring for ivy.assert_supports_inplace also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array

        Returns
        -------
        ret
            True if support, raises exception otherwise

        """
        return ivy.assert_supports_inplace(self)

    def is_ivy_array(self: ivy.Array, exclusive: Optional[bool] = False) -> bool:
        """
        ivy.Array instance method variant of ivy.is_ivy_array. This method simply wraps the
        function, and so the docstring for ivy.is_ivy_array also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.

        Returns
        -------
        ret
            Boolean, whether or not x is an ivy array.

        """
        return ivy.is_ivy_array(self, exclusive)

    def copy_array(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.copy_array. This method simply wraps the
        function, and so the docstring for ivy.copy_array also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a copy of the input array ``x``.

        """
        return ivy.copy_array(self, out=out)

    def to_scalar(self: ivy.Array) -> Number:
        """
        ivy.Array instance method variant of ivy.to_scalar. This method simply wraps
        the function, and so the docstring for ivy.to_scalar also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.

        Returns
        -------
        ret
            a scalar copying the element of the array ``x``.

        """
        return ivy.to_scalar(self)

    def floormod(self: ivy.Array, x: Union[ivy.Array, ivy.NativeArray], \
                out: Optional[Union[ivy.Array, ivy.NativeArray]] = None) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.floormod. This method simply wraps the
        function, and so the docstring for ivy.floormod also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        x
            input array for the denominator
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            An array of the same shape and type as x, with the elements floor modded.

        """
        return ivy.floormod(self, x, out=out)

    def fourier_encode(self: ivy.Array, max_freq: Union[float, ivy.Array, ivy.NativeArray], \
                        num_bands: int = 4, linear: bool = False, concat: bool = True, 
                        flatten: bool = False) -> Union[ivy.Array, ivy.NativeArray, Tuple]:
        """
        ivy.Array instance method variant of ivy.fourier_encode. This method simply wraps the
        function, and so the docstring for ivy.fourier_encode also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array to encode
        max_freq
            The maximum frequency of the encoding.
        num_bands
            The number of frequency bands for the encoding. Default is 4.
        linear
            Whether to space the frequency bands linearly as opposed to geometrically.
            Default is False.
        concat
            Whether to concatenate the position, sin and cos values, or return seperately.
            Default is True.
        flatten
            Whether to flatten the position dimension into the batch dimension. Default is
            False.

        Returns
        -------
        ret
            New array with the final dimension expanded, and the encodings stored in this
            channel.

        """
        return ivy.fourier_encode(self, max_freq, num_bands, linear, concat, flatten)

    def value_is_nan(self: ivy.Array, include_infs: Optional[bool] = True) -> bool:
        """
        ivy.Array instance method variant of ivy.value_is_nan. This method simply wraps the
        function, and so the docstring for ivy.value_is_nan also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        include_infs
            Whether to include infs and -infs in the check. Default is True.

        Returns
        -------
        ret
            Boolean as to whether the input value is a nan or not.

        """
        return ivy.value_is_nan(self, include_infs)
