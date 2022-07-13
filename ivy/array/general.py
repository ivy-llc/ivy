# global
import abc
from typing import Any, Iterable, Union, Optional

# ToDo: implement all methods here as public instance methods

# local
import ivy


class ArrayWithGeneral(abc.ABC):
    def all_equal(self: ivy.Array, x2: Iterable[Any], equality_matrix: bool = False):
        """
        ivy.Container instance method variant of ivy.all_equal.
        This method simply wraps the function, and so the docstring for
        ivy.all_equal also applies to this method with minimal changes.

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
        ivy.Container instance method variant of ivy.to_numpy.
        This method simply wraps the function, and so the docstring for
        ivy.to_numpy also applies to this method with minimal changes.

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
