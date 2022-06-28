# global
import abc
# ToDo: implement all methods here as public instance methods

# local
import ivy


class ArrayWithGeneral(abc.ABC):
    def all_equal(
            self,
            x,
            equality_matrix: bool = False
    ):
        """
        ivy.Container static method variant of ivy.all_equal. This method simply wraps the
        function, and so the docstring for ivy.all_equal also applies to this method
        with minimal changes.

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
        return ivy.all_equal(self, x, equality_matrix)
