# global
import abc
from typing import Union, Optional

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithGradients(abc.ABC):
    def variable(self: ivy.Array) -> ivy.Variable:
        """
        ivy.Array instance method variant of ivy.variable. This method simply wraps
        the function, and so the docstring for ivy.variable also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            An ivy array.

        Returns
        -------
        ret
            An ivy variable that supports gradient computation.

        Examples
        --------
        With :code:`ivy.Array` input:

        >>> ivy.set_backend("tensorflow")
        >>> x = ivy.array([2., 4., -1.])
        >>> y = x.variable()
        >>> y
        ivy.array([ 2.,  4., -1.])
        >>> ivy.unset_backend()
        """
        return ivy.variable(self)

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

    def variable_data(self: ivy.Array) -> bool:
        """
        ivy.Array instance method variant of ivy.variable_data. This method simply wraps
        the function, and so the docstring for ivy.variable_data also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            An ivy variable.

        Returns
        -------
        ret
            The internal data stored by the variable

        """
        return ivy.variable_data(self)

    def stop_gradient(
        self: ivy.Array,
        /,
        *,
        preserve_type: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.stop_gradient. This method simply
        wraps the function, and so the docstring for ivy.stop_gradient also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Array for which to stop the gradient.
        preserve_type
            Whether to preserve the input type (ivy.Variable or ivy.Array),
            otherwise an array is always returned. Default is True.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The same array x, but with no gradient information.
        """
        return ivy.stop_gradient(self, preserve_type=preserve_type, out=out)

    def adam_step(
        self: ivy.Array,
        mw: Union[ivy.Array, ivy.NativeArray],
        vw: Union[ivy.Array, ivy.NativeArray],
        step: Union[int, float],
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        out: Optional[ivy.Array] = None,
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
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

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
        return ivy.adam_step(
            self, mw, vw, step, beta1=beta1, beta2=beta2, epsilon=epsilon, out=out
        )

    def optimizer_update(
        self: ivy.Array,
        effective_grad: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        stop_gradients: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.optimizer_update. This method simply
        wraps the function, and so the docstring for ivy.optimizer_update also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        effective_grad
            Effective gradients of the cost c with respect to the weights ws,
            [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be updated
            relative to the gradient.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the optimizer updates.

        """
        return ivy.optimizer_update(
            self, effective_grad, lr, stop_gradients=stop_gradients, out=out
        )

    def gradient_descent_update(
        self: ivy.Array,
        dcdw: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        stop_gradients: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gradient_descent_update.
        This method simply wraps the function, and so the docstring for
        ivy.gradient_descent_update also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be
            updated relative to the gradient.
        inplace
            Whether to perform the operation inplace, for backends which support inplace
            variable updates, and handle gradients behind the scenes such as PyTorch.
            If the update step should form part of a computation graph
            (i.e. higher order optimization), then this should be set to False.
            Default is True, provided the backend framework supports it.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.

        Returns
        -------
        ret
            The new weights, following the gradient descent updates.

        Examples
        --------
        With :code: `ivy.Array` inputs:

        >>> w = ivy.array([[[5., 3., 2.], [0., 4., 1.], [-2., 3., -1.]]])
        >>> dcdw = ivy.array([[[0.5, 0.92, 0.1], [0.2, 0.7, 0.3], [0.3, 0.8, 0.01]]])
        >>> lr = ivy.array(0.3)
        >>> NewWeights = w.gradient_descent_update(dcdw, lr, inplace=False)
        >>> print(NewWeights)
            ivy.array([[[ 4.85,  2.72,  1.97],
                        [-0.06,  3.79,  0.91],
                        [-2.09,  2.76, -1.  ]]])
        """
        return ivy.gradient_descent_update(
            self, dcdw, lr, stop_gradients=stop_gradients, out=out
        )

    def lars_update(
        self: ivy.Array,
        dcdw: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        decay_lambda: float = 0,
        stop_gradients: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.lars_update. This method simply
        wraps the function, and so the docstring for ivy.lars_update also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate, the rate at which the weights should be updated relative to
            the gradient.
        decay_lambda
            The factor used for weight decay. Default is zero.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the LARS updates.

        """
        return ivy.lars_update(
            self,
            dcdw,
            lr,
            decay_lambda=decay_lambda,
            stop_gradients=stop_gradients,
            out=out,
        )

    def adam_update(
        self: ivy.Array,
        dcdw: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        mw_tm1: Union[ivy.Array, ivy.NativeArray],
        vw_tm1: Union[ivy.Array, ivy.NativeArray],
        step: int,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        stop_gradients: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.adam_update. This method simply
        wraps the function, and so the docstring for ivy.adam_update also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be updated
            relative to the gradient.
        mw_tm1
            running average of the gradients, from the previous time-step.
        vw_tm1
            running average of second moments of the gradients, from the previous
            time-step.
        step
            training step.
        beta1
            gradient forgetting factor (Default value = 0.9).
        beta2
            second moment of gradient forgetting factor (Default value = 0.999).
        epsilon
            divisor during adam update, preventing division by zero
            (Default value = 1e-7).
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, and also new mw and vw, following the adam
            updates.

        """
        return ivy.adam_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            stop_gradients=stop_gradients,
            out=out,
        )

    def lamb_update(
        self: ivy.Array,
        dcdw: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        mw_tm1: Union[ivy.Array, ivy.NativeArray],
        vw_tm1: Union[ivy.Array, ivy.NativeArray],
        step: int,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        max_trust_ratio: Union[int, float] = 10,
        decay_lambda: float = 0,
        stop_gradients: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.lamb_update. This method simply
        wraps the function, and so the docstring for ivy.lamb_update also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be updated
            relative to the gradient.
        mw_tm1
            running average of the gradients, from the previous time-step.
        vw_tm1
            running average of second moments of the gradients, from the previous
            time-step.
        step
            training step.
        beta1
            gradient forgetting factor (Default value = 0.9).
        beta2
            second moment of gradient forgetting factor (Default value = 0.999).
        epsilon
            divisor during adam update, preventing division by zero
            (Default value = 1e-7).
        max_trust_ratio
            The maximum value for the trust ratio. Default is 10.
        decay_lambda
            The factor used for weight decay. Default is zero.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the LAMB updates.

        """
        return ivy.lamb_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_trust_ratio=max_trust_ratio,
            decay_lambda=decay_lambda,
            stop_gradients=stop_gradients,
            out=out,
        )
