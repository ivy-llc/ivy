# global
import abc
from numbers import Number
from typing import Any, Iterable, Union, Optional

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

    def stable_pow(
        self: ivy.Array,
        exponent: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container],
        min_base: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.stable_pow. This method simply wraps
        the function, and so the docstring for ivy.stable_pow also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array, the base of the exponentiation.
        exponent
            The exponent of the exponentiation.
        min_base
            The minimum base to use, use global ivy._MIN_BASE by default.

        Returns
        -------
        ret
            Numpy array containing the elements of base raised to the power of the
            corresponding element of exponent.

        Examples
        --------

        >>> x = ivy.asarray([2, 4, 5])
        >>> y = x.stable_pow(2)
        >>> print(y)
        ivy.array([ 4, 16, 25])

        >>> x = ivy.asarray([2, 4, 5])
        >>> y = ivy.Array.stable_pow(x, 4, min_base=1)
        >>> print(y)
        ivy.array([  81,  625, 1300])

        >>> x = ivy.asarray([[4, 5, 6], [7, 8, 9]])
        >>> y = ivy.asarray([[1, 2,3], [2, 3, 4]])
        >>> z = ivy.Array.stable_pow(x, y)
        >>> print(z)
        ivy.array([[   4,   25,  216],
               [  49,  512, 6560]])
        """
        return ivy.stable_pow(self, exponent, min_base=min_base)

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
