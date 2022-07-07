# global
import abc
from typing import Tuple

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithSet(abc.ABC):
    def unique_counts(
        self: ivy.Array
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.unique_counts. This method simply 
        wraps the function, and so the docstring for ivy.unique_counts also applies 
        to this method with minimal changes.

        Examples
        --------
        With :code:`ivy.Array` instance method:

        >>> x = ivy.array([0., 1., 2. , 1. , 0.])
        >>> y = x.unique_counts()
        >>> print(y)
        uc(values=ivy.array([0.,1.,2.]),counts=ivy.array([2,2,1])) 
        """
        return ivy.unique_counts(self._data)
