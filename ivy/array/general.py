# global
import abc
import numpy as np
from numbers import Number
from typing import Any, Iterable, Union, Optional, Dict, Callable, List, Tuple

# ToDo: implement all methods here as public instance methods

# local
import ivy


class ArrayWithGeneral(abc.ABC):
    def is_native_array(
        self: ivy.Array, *, exclusive: bool = False, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.is_native_array. This method simply
        wraps the function, and so the docstring for ivy.is_native_array
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.

        Returns
        -------
        ret
            Boolean, whether or not x is a native array.
        """
        return ivy.is_native_array(self._data, exclusive=exclusive, out=out)

    def is_ivy_array(
        self: ivy.Array, *, exclusive: bool = False, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.is_ivy_array. This method simply
        wraps the function, and so the docstring for ivy.is_ivy_array also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.
        """
        return ivy.is_ivy_array(self._data, exclusive=exclusive, out=out)

    def is_array(
        self: ivy.Array, *, exclusive: bool = False, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.is_array. This method simply wraps the
        function, and so the docstring for ivy.is_array also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.
        """
        return ivy.is_array(self._data, exclusive=exclusive, out=out)

    def is_ivy_container(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.is_ivy_container. This method simply
        wraps the function, and so the docstring for ivy.is_ivy_container also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check

        Returns
        -------
        ret
            Boolean, whether or not x is an ivy container.
        """
        return ivy.is_ivy_container(self._data, out=out)

    def all_equal(
        self: ivy.Array, x2: Iterable[Any], equality_matrix: bool = False
    ) -> Union[bool, ivy.Array, ivy.NativeArray]:
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

    def has_nans(self: ivy.Array, include_infs: bool = True):
        """
        ivy.Array instance method variant of ivy.has_nans. This method simply wraps the
        function, and so the docstring for ivy.has_nans also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        include_infs
            Whether to include ``+infinity`` and ``-infinity`` in the check.
            Default is True.

        Returns
        -------
        ret
            Boolean as to whether the array contains nans.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = x.has_nans()
        >>> print(y)
        False
        """
        return ivy.has_nans(self, include_infs)

    def unstack(self: ivy.Array, axis: int, /, *, keepdims: bool = False) -> ivy.Array:
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

        Examples
        --------
        >>> x = ivy.array([[1, 2], [3, 4]])
        >>> y = x.unstack(axis=0)
        >>> print(y)
        [ivy.array([1, 2]), ivy.array([3, 4])]

        >>> x = ivy.array([[1, 2], [3, 4]])
        >>> y = x.unstack(axis=1, keepdims=True)
        >>> print(y)
        [ivy.array([[1],
                [3]]), ivy.array([[2],
                [4]])]
        """
        return ivy.unstack(self._data, axis, keepdims=keepdims)

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

    def scatter_nd(
        self: ivy.Array,
        updates: Union[ivy.Array, ivy.NativeArray],
        shape: Optional[ivy.Array] = None,
        reduction: str = "sum",
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Scatter updates into an array according to indices.

        Parameters
        ----------
        self
            The tensor in which to scatter the results
        indices
            Tensor of indices
        updates
            values to update input tensor with
        shape
            The shape of the result. Default is None, in which case tensor argument must
            be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            New array of given shape, with the values scattered at the indices.

        Examples
        --------
        scatter values into an array

        >> arr = ivy.array([1,2,3,4,5,6,7,8, 9, 10])
        >> indices = ivy.array([[4], [3], [1], [7]])
        >> updates = ivy.array([9, 10, 11, 12])
        >> scatter = indices.scatter_nd(updates, tensor=arr, reduction='replace')
        >> print(scatter)
        ivy.array([ 1, 11,  3, 10,  9,  6,  7, 12,  9, 10])

        scatter values into an empty array

        >> shape = ivy.array([2, 5])
        >> indices = ivy.array([[1,4], [0,3], [1,1], [0,2]])
        >> updates = ivy.array([25, 40, 21, 22])
        >> scatter = indices.scatter_nd(updates, shape=shape)
        >> print(scatter)
        ivy.array([[ 0,  0, 22, 40,  0],
                    [ 0, 21,  0,  0, 25]])
        """
        return ivy.scatter_nd(self, updates, shape, reduction, out=out)

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

        Examples
        --------
        >> x = ivy.array([[[5,4],
                       [11, 2]],
                      [[3, 5],
                       [9, 7]]])
        >> reduced = x.einops_reduce('a b c -> b c', 'max')
        >> print(reduced)
        ivy.array([[ 5,  5],
                   [11,  7]])

        >> x = ivy.array([[[5, 4, 3],
                        [11, 2, 9]],
                       [[3, 5, 7],
                        [9, 7, 1]]])
        >> reduced = x.einops_reduce('a b c -> a () c', 'min')
        >> print(reduced)
        ivy.array([[[5, 2, 3]],
                   [[3, 5, 1]]])
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

        Examples
        --------
        >> x = ivy.array([5,4])
        >> repeated = x.einops_repeat('a -> a c', c=3)
        >> print(repeated)
        ivy.array([[5, 4],
                   [5, 4],
                  [5, 4]])

        >> x = ivy.array([[5,4],
                    [2, 3]])
        >> repeated = x.einops_repeat('a b ->  a b c', c=3)
        >> print(repeated)
        ivy.array([[[5, 5, 5],
                    [4, 4, 4]],
                   [[2, 2, 2],
                    [3, 3, 3]]])

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

    def supports_inplace_updates(self: ivy.Array) -> bool:
        """
        ivy.Array instance method variant of ivy.supports_inplace_updates. This method
        simply wraps the function, and so the docstring for ivy.supports_inplace also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input array whose elements' data type is to be checked.

        Returns
        -------
        ret
            Bool value depends on whether the currently active backend
            framework supports in-place operations with argument's data type.

        Examples
        --------
        With `ivy.Array` input and backend set as "tensorflow":
        >>> x = ivy.array([1., 4.2, 2.2])
        >>> ret = x.supports_inplace()
        >>> print(ret)
        False
        """
        return ivy.supports_inplace_updates(self)

    def inplace_decrement(
        self: Union[ivy.Array, ivy.NativeArray], val: Union[ivy.Array, ivy.NativeArray]
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.inplace_decrement. This method simply
        wraps the function, and so the docstring for ivy.inplace_decrement also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input array to be decremented by the defined value.
        val
            The value of decrement.

        Returns
        -------
        ret
            The array following an in-place decrement.

        Examples
        --------
        With :code:`ivy.Array` instance methods:

        >>> x = ivy.array([5.7, 4.3, 2.5, 1.9])
        >>> y = x.inplace_decrement(1)
        >>> print(y)
        ivy.array([4.7, 3.3, 1.5, 0.9])

        >>> x = ivy.asarray([4., 5., 6.])
        >>> y = x.inplace_decrement(2.5)
        >>> print(y)
        ivy.array([1.5, 2.5, 3.5])

        """
        return ivy.inplace_decrement(self, val)

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

        """
        return ivy.clip_vector_norm(self, max_norm, p=p, out=out)

    def array_equal(self: ivy.Array, x: Union[ivy.Array, ivy.NativeArray]) -> bool:
        """
        ivy.Array instance method variant of ivy.array_equal. This method simply wraps
        the function, and so the docstring for ivy.array_equal also applies to this
        method with minimal changes.

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

    def arrays_equal(
        self: ivy.Array, x: List[Union[ivy.Array, ivy.NativeArray]]
    ) -> bool:
        """
        ivy.Array instance method variant of ivy.arrays_equal. This method simply wraps
        the function, and so the docstring for ivy.arrays_equal also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array
        x
            input list of arrays to compare to ``self``

        Returns
        -------
        ret
            Boolean, whether the input arrays are equal

        """
        return ivy.arrays_equal([self] + x)

    def assert_supports_inplace(self: ivy.Array) -> bool:
        """
        ivy.Array instance method variant of ivy.assert_supports_inplace. This method
        simply wraps the function, and so the docstring for ivy.assert_supports_inplace
        also applies to this method with minimal changes.

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

    def copy_array(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.copy_array. This method simply wraps
        the function, and so the docstring for ivy.copy_array also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

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

    def fourier_encode(
        self: ivy.Array,
        max_freq: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        num_bands: int = 4,
        linear: bool = False,
        concat: bool = True,
        flatten: bool = False,
    ) -> Union[ivy.Array, ivy.NativeArray, Tuple]:
        """
        ivy.Array instance method variant of ivy.fourier_encode. This method simply
        wraps the function, and so the docstring for ivy.fourier_encode also applies to
        this method with minimal changes.

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
            Whether to concatenate the position, sin and cos values, or return
            seperately. Default is True.
        flatten
            Whether to flatten the position dimension into the batch dimension.
            Default is False.

        Returns
        -------
        ret
            New array with the final dimension expanded, and the encodings stored in
            this channel.

        """
        return ivy.fourier_encode(self, max_freq, num_bands, linear, concat, flatten)

    def value_is_nan(self: ivy.Array, include_infs: Optional[bool] = True) -> bool:
        """
        ivy.Array instance method variant of ivy.value_is_nan. This method simply wraps
        the function, and so the docstring for ivy.value_is_nan also applies to this
        method with minimal changes.

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

    def exists(self: ivy.Array) -> bool:
        """
        ivy.Array instance method variant of ivy.exists. This method simply wraps
        the function, and so the docstring for ivy.exists also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.

        Returns
        -------
        ret
            True if x is not None, else False.

        """
        return ivy.exists(self)

    def default(
        self: ivy.Array,
        default_val: Union[ivy.Array, ivy.NativeArray],
        catch_exceptions: bool = False,
        rev: bool = False,
        with_callable: bool = False,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.default. This method simply wraps the
        function, and so the docstring for ivy.default also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        default_val
            The default value.
        catch_exceptions
            Whether to catch exceptions from callable x. Default is False.
        rev
            Whether to reverse the input x and default_val. Default is False.
        with_callable
            Whether either of the arguments might be callable functions.
            Default is False.

        Returns
        -------
        ret
            x if x exists (is not None), else default.

        """
        return ivy.default(self, default_val, catch_exceptions, rev, with_callable)

    def stable_pow(
        self: ivy.Array,
        exponent: Union[Number, ivy.Array, ivy.NativeArray],
        min_base: float = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.stable_pow. This method simply wraps
        the function, and so the docstring for ivy.stable_pow also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array, used as the base.
        exponent
            The exponent number.
        min_base
            The minimum base to use, use global ivy._MIN_BASE by default.

        Returns
        -------
        ret
            The new item following the numerically stable power.

        """
        return ivy.stable_pow(self, exponent, min_base=min_base)

    def inplace_update(
        self: ivy.Array,
        val: Union[ivy.Array, ivy.NativeArray],
        ensure_in_backend: bool = False,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.inplace_update. This method simply
        wraps the function, and so the docstring for ivy.inplace_update also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array to update
        val
            The array to update the variable with.
        ensure_in_backend
            Whether or not to ensure that the `ivy.NativeArray` is also inplace updated.
            In cases where it should be, backends which do not natively support inplace
            updates will raise an exception.

        Returns
        -------
        ret
            The array following the in-place update.

        """
        return ivy.inplace_update(self, val, ensure_in_backend=ensure_in_backend)

    def inplace_increment(
        self: ivy.Array, val: Union[ivy.Array, ivy.NativeArray]
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.inplace_increment. This
        method wraps the function, and so the docstring for
        ivy.inplace_increment also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input array to be incremented by the defined value.
        val
            The value of increment.

        Returns
        -------
        ret
            The array following an in-place increment.

        Examples
        --------
        With :code:`ivy.Array` instance methods:

        >>> x = ivy.array([5.7, 4.3, 2.5, 1.9])
        >>> y = x.inplace_increment(1)
        >>> print(y)
        ivy.array([6.7, 5.3, 3.5, 2.9])

        >>> x = ivy.asarray([4., 5., 6.])
        >>> y = x.inplace_increment(2.5)
        >>> print(y)
        ivy.array([6.5, 7.5, 8.5])

        """
        return ivy.inplace_increment(self, val)

    def clip_matrix_norm(
        self: ivy.Array,
        max_norm: float,
        /,
        *,
        p: float = 2.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.clip_matrix_norm. This method simply
        wraps the function, and so the docstring for ivy.clip_matrix_norm also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array
        max_norm
            The maximum value of the array norm.
        p
            The p-value for computing the p-norm. Default is 2.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array with the matrix norm downscaled to the max norm if needed.

        Examples
        --------
        With :code:`ivy.Array` instance method:

        >>> x = ivy.array([[0., 1., 2.]])
        >>> y = x.clip_matrix_norm(2.0)
        >>> print(y)
        ivy.array([[0.   , 0.894, 1.79 ]])

        """
        return ivy.clip_matrix_norm(self, max_norm, p, out=out)

    def scatter_flat(
        self: ivy.Array,
        updates: Union[ivy.Array, ivy.NativeArray],
        size: Optional[int] = None,
        reduction: str = "sum",
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.scatter_flat. This method simply wraps
        the function, and so the docstring for ivy.scatter_flat also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array containing the indices where the new values will occupy
        updates
            Values for the new array to hold.
        size
            The size of the result.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max' or
            'replace'
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as
            updates if None.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            New array of given shape, with the values scattered at the indices.

        """
        return ivy.scatter_flat(self, updates, size=size, reduction=reduction, out=out)

    def indices_where(
        self: ivy.Array, *, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.indices_where. This method simply
        wraps the function, and so the docstring for ivy.indices_where also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array for which indices are desired
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Indices for where the boolean array is True.

        """
        return ivy.indices_where(self, out=out)

    def one_hot(
        self: ivy.Array,
        depth: int,
        *,
        device: Union[ivy.Device, ivy.NativeDevice] = None,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.one_hot. This method simply wraps the
        function, and so the docstring for ivy.one_hot also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array containing the indices for which the ones should be scattered
        depth
            Scalar defining the depth of the one-hot dimension.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            Same as x if None.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Tensor of zeros with the same shape and type as a, unless dtype provided
            which overrides.

        """
        return ivy.one_hot(self, depth, device=device, out=out)

    def get_num_dims(self: ivy.Array, as_array: bool = False) -> int:
        """
        ivy.Array instance method variant of ivy.shape. This method simply wraps the
        function, and so the docstring for ivy.shape also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array to infer the number of dimensions  for
        as_array
            Whether to return the shape as a array, default False.

        Returns
        -------
        ret
            Shape of the array

        """
        return ivy.get_num_dims(self, as_array)
