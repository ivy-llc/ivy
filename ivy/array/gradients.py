# global
import abc
from typing import Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithGradients(abc.ABC):
    def is_variable(self: ivy.Array, exclusive: bool = False) -> bool:
        """
        ivy.Array instance method variant of ivy.is_variable. This method simply wraps
        the function, and so the docstring for ivy.is_variable also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            An ivy array.
        exclusive
            Whether to check if the data type is exclusively a variable, rather than an
            array. For frameworks like JAX that do not have exclusive variable types,
            the function will always return False if this flag is set, otherwise the
            check is the same for general arrays. Default is False.

        Returns
        -------
        ret
            Boolean, true if ``self`` is a trainable variable, false otherwise.

        Examples
        --------
        With :code:`ivy.Array` input:

        >>> x = ivy.array([-2, 0.4, 7])
        >>> is_var = x.is_variable(True)
        >>> print(is_var)
            False
        """
        return ivy.is_variable(self, exclusive)

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

        Parameters
        ----------
        self
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        mw
            running average of the gradients.
        vw
            running average of second moments of the gradients.
        step
            training step.
        beta1
            gradient forgetting factor (Default value = 0.9).
        beta2
            second moment of gradient forgetting factor (Default value = 0.999).
        epsilon
            divisor during adam update, preventing division by zero
            (Default value = 1e-7).

        Returns
        -------
        ret
            The adam step delta.

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

    def lars_update( 
    self: ivy.Array,
    w: Union[ivy.Array, ivy.NativeArray], 
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[int, float],
    lamda: Union[int, float],
    ) -> ivy.Array:
     """
        ivy.Array instance method variant of ivy.lars_update. This method simply wraps the
        function, and so the docstring for ivy.lars_update also applies to this method
        with minimal changes.

       
        Examples
        --------
        With :code:`ivy.Array` inputs:

        >>> w = ivy.array([[0.], [0.], [0.]])
        >>> dcdw = ivy.array([[[1.1], [3.2], [-6.3]]])
        >>> lr = ivy.array(3)
        >>> lambda = ivy.array(0.1)
        >>> lars_update_delta = w.lars_update(dcdw, lr, lamda)
        >>> print(lars_update_delta)
            (ivy.array([[[ 0.639], [ 0.639], [-0.639]]]),
            ivy.array([[[ 0.11], [ 0.32], [-0.63]]]))
        """
     return ivy.lars_update(self, w, dcdw, lr, lamda)