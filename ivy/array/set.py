# global
import abc
from typing import Optional, Tuple

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithSet(abc.ABC):
    def unique_counts(
        self: ivy.Array,
        *,
        out: Optional[Tuple[ivy.Array, ivy.Array]] = None
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_counts. This method simply wraps the
        function, and so the docstring for ivy.unique_counts also applies to this method
        with minimal changes.

        Examples
        --------
        With :code:`ivy.Array` instance method:

        >>> x = ivy.array([0., 1., 3. , 2. , 1. , 0.])
        >>> y = x.unique_counts()
        >>> print(y)
        Tuple([0., 1., 2., 3.],[2,2,1,1])
        """
        return ivy.unique_counts(self._data, out=out)
