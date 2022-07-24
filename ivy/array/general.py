# global
import abc
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

    def floormod(
        self: ivy.Array,
        y: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        """
        ivy.Array instance method variant of ivy.gather_nd. This method simply wraps the
        function, and so the docstring for ivy.gather_nd also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            array, input to floormod
        y
            array, denominator input for floormod.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array of the same shape and type as x, with the elements floor modded.

        Examples
        --------
        >>> x = ivy.array([9, 8, 7])
        >>> y = ivy.array([2, 3, 4])
        >>> print(x.floormod(y))
        ivy.array([1, 2, 3])

        >>> a = ivy.array([3, 5, 8])
        >>> b = ivy.array([1, 2, 3])
        >>> c = ivy.array([0, 0, 0])
        >>> a.floormod(b, out=c)
        >>> print(c)
        ivy.array([0, 1, 2])

        >>> i = ivy.array([7, 6, 4])
        >>> j = ivy.array([4, 5, 4])
        >>> i.floormod(j, out=i)
        >>> print(i)
        ivy.array([3, 1, 0])

        """
        return ivy.floormod(self, y, out=out)

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
