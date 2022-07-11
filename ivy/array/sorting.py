# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithSorting(abc.ABC):
    def sort(
            self,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True,
            *,
            out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.sort also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([7, 8, 6])
        >>> y = ivy.sort(x)
        >>> print(y)
        ivy.array([6, 7, 8])

         >>> x = ivy.array([8.5, 8.2, 7.6])
        >>> y = ivy.sort(x)
        >>> print(y)
        ivy.array([7.6, 8.2, 8.5])

        With a mix of :code:`ivy.Container` and :code:`ivy.Array` input:

        >>>x = ivy.Container(a=ivy.native_array([8, 0.5, 6]),\
                            b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>>y = ivy.sort(x)
        >>>print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.], \
                          [0., 0.4]])
        }

        >>>x = ivy.Container(a=ivy.native_array([1, 0.3, 7]),\
                                b=ivy.array([[2, 0.2], [6, 5]]))
        >>>y = ivy.sort(x)
        >>>print(y)
        {
            a: ivy.array([0.3, 1., 7.]),
            b: ivy.array([[0.2, 2.], \
                          [5., 6.]])
        }

        """
        return ivy.sort(self._data)

