# global
import abc
from typing import Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithGradients(abc.ABC):
    def adam_step(self: ivy.Array,
                  mw: Union[ivy.Array, ivy.NativeArray],
                  vw: Union[ivy.Array, ivy.NativeArray],
                  step: Union[int, float],
                  beta1=0.9,
                  beta2=0.999,
                  epsilon=1e-7,
                  ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.adam_step. This method simply wraps the
        function, and so the docstring for ivy.adam_step also applies to this method
        with minimal changes.

        Examples
        --------
        With :code:`ivy.Array` inputs:
        >>> dcdw = ivy.array([1, 2, 3])
        >>> mw = ivy.zeros(3)
        >>> vw = ivy.zeros(1)
        >>> step = ivy.array(3)
        >>> adam_step_delta = dcdw.adam_step(dcdw, mw, vw, step)
        >>> print(adam_step_delta)
            (ivy.array([0.639, 0.639, 0.639]),
            ivy.array([0.1, 0.2, 0.3]),
            ivy.array([0.001, 0.004, 0.009]))
        """
        return ivy.adam_step(self._data, mw, vw, step, beta1, beta2, epsilon)
