# global
import abc
from typing import Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithGradients(abc.ABC):
    def adam_step(
        self: ivy.Array,
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
        >>> dcdw = ivy.array([[[1.1], [3.2], [-6.3]]])
        >>> mw = ivy.array([[0.], [0.], [0.]])
        >>> vw = ivy.array([[0.], [0.], [0.]])
        >>> step = ivy.array(3)
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step)
        >>> print(adam_step_delta)
            (ivy.array([[[ 0.639], [ 0.639], [-0.639]]]),
            ivy.array([[[ 0.11], [ 0.32], [-0.63]]]),
            ivy.array([[[0.00121], [0.0102 ], [0.0397 ]]]))
        """
        return ivy.adam_step(self, mw, vw, step, beta1, beta2, epsilon)
