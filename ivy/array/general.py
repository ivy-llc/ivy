# global
import abc
# ToDo: implement all methods here as public instance methods

# local
import ivy


class ArrayWithGeneral(abc.ABC):
    def all_equal(
            self,
            equality_matrix: bool = False
    ):
        """Determines whether the inputs are all equal.

        Parameters
        ----------
        xs
            inputs to compare.
        equality_matrix
            Whether to return a matrix of equalities comparing each
            input with every other.
            Default is False.

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix array of booleans
            if equality_matrix=True is set.

        Examples
        --------

        With :code:`ivy.Array` input:

        >>> x1 = ivy.array([1, 2, 3])
        >>> x2 = ivy.array([1, 0, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        False

        >>> x1 = ivy.array([1, 0, 1, 1])
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        True

        With :code:`ivy.NativeArray` input:

        >>> x1 = ivy.native_array([1, 2, 3])
        >>> x2 = ivy.native_array([1, 0, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        False

        >>> x1 = ivy.native_array([1, 0, 1, 1])
        >>> x2 = ivy.native_array([1, 0, 1, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        True

        With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` input:

        >>> x1 = ivy.array([1, 1, 0, 1.2, 1])
        >>> x2 = ivy.native_array([1, 1, 0, 0.5, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        False

        >>> x1 = ivy.array([1, 1, 0, 0.5, 1])
        >>> x2 = ivy.native_array([1, 1, 0, 0.5, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        True

        """
        return ivy.all_equal(self, equality_matrix)
