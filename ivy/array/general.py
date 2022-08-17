# global
import abc
from numbers import Number
from typing import Any, Iterable, Union, Optional, Dict, Callable

# ToDo: implement all methods here as public instance methods

# local
import ivy


class ArrayWithGeneral(abc.ABC):
    def all_equal(self: ivy.Array, x2: Iterable[Any], equality_matrix: bool = False):
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

        Examples
        --------
        With :code:`ivy.Array` instance method:

        >>> x1 = ivy.array([1, 2, 3])
        >>> x2 = ivy.array([1, 0, 1])
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        False

        With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` instance method:

        >>> x1 = ivy.array([1, 1, 0, 0.5, 1])
        >>> x2 = ivy.native_array([1, 1, 0, 0.5, 1])
        >>> y = x1.all_equal(x2, equality_matrix= True)
        >>> print(y)
        ivy.array([[ True,  True], [ True,  True]])

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
        return ivy.unstack(self._data, axis, keepdims)

    def cumprod(
        self: ivy.Array,
        axis: int = 0,
        exclusive: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cumprod. This method simply wraps the
        function, and so the docstring for ivy.cumprod also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array
        axis
            int, axis along which to take the cumulative product. Default is 0.
        exclusive
            optional bool, whether to exclude the first value of the input array.
            Default is False.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Input array with cumulatively multiplied elements along the specified axis.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3, 4, 5])
        >>> y = x.cumprod()
        >>> print(y)
        ivy.array([  1,   2,   6,  24, 120])

        >>> x = ivy.array([[2, 3], [5, 7], [11, 13]])
        >>> y = ivy.zeros((3, 2))
        >>> x.cumprod(axis=1, exclusive=True, out=y)
        >>> print(y)
        ivy.array([[ 1.,  2.],
                   [ 1.,  5.],
                   [ 1., 11.]])
        """
        return ivy.cumprod(self._data, axis, exclusive, out=out)

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
        tensor: Union[ivy.Array, ivy.NativeArray] = None,
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

        >> arr = ivy.Array([1,2,3,4,5,6,7,8, 9, 10])
        >> indices = ivy.array([[4], [3], [1], [7]])
        >> updates = ivy.array([9, 10, 11, 12])
        >> scatter = indices.scatter_nd(updates, tensor=arr, reduction='replace')
        >> print(scatter)
        ivy.array([ 1, 11,  3, 10,  9,  6,  7, 12,  9, 10])

        scatter values into an empty array

        >> shape = ivy.array([8])
        >> indices = ivy.array([[4], [3], [1], [7]])
        >> updates = ivy.array([9, 10, 11, 12])
        >> scatter = indices.scatter_nd(updates, shape=shape)
        >> print(scatter)
        ivy.array([ 0, 11,  0, 10,  9,  0,  0, 12])
        """
        return ivy.scatter_nd(self, updates, shape, reduction, tensor=tensor, out=out)
    
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

    def to_numpy(self: ivy.Array):
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

        Examples
        --------
        With :code:`ivy.Array` instance methods:

        >>> x = ivy.array([1, 0, 1, 1])
        >>> y = x.to_numpy()
        >>> print(y)
        [1 0 1 1]

        >>> x = ivy.array([1, 0, 0, 1])
        >>> y = x.to_numpy()
        >>> print(y)
        [1 0 0 1]

        """
        return ivy.to_numpy(self)

    def to_list(self: ivy.Array):
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

    def clip_matrix_norm(
        self: ivy.Array,
        max_norm: float,
        p: float = 2.0,
        *,
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
