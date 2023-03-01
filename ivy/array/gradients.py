# global
import abc
from typing import Union, Optional

# local
import ivy

# ToDo: implement all methods here as public instance methods


class _ArrayWithGradients(abc.ABC):
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
            Whether to preserve gradient computation on ivy.Array instances. Default is
            True.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The same array x, but with no gradient information.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> y = x.stop_gradient(preserve_type=True)
        >>> print(y)
        ivy.array([1., 2., 3.])

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
        With :class:`ivy.Array` inputs:

        >>> dcdw = ivy.array([1, 2, 3])
        >>> mw = ivy.ones(3)
        >>> vw = ivy.ones(1)
        >>> step = ivy.array(3)
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step)
        >>> print(adam_step_delta)
        (ivy.array([0.2020105,0.22187898,0.24144873]),
            ivy.array([1.,1.10000002,1.20000005]),
            ivy.array([1.,1.00300002,1.00800002]))

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
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the optimizer updates.

        Examples
        --------
        >>> w = ivy.array([1., 2., 3.])
        >>> effective_grad = ivy.zeros(3)
        >>> lr = 3e-4
        >>> ws_new = w.optimizer_update(effective_grad, lr)
        >>> print(ws_new)
        ivy.array([1., 2., 3.])

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
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new weights, following the gradient descent updates.

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> w = ivy.array([[1., 2, 3],
        ...                [4, 6, 1],
        ...                [1, 0, 7]])
        >>> dcdw = ivy.array([[0.5, 0.2, 0.1],
        ...                   [0.3, 0.6, 0.4],
        ...                   [0.4, 0.7, 0.2]])
        >>> lr = ivy.array(0.1)
        >>> new_weights = w.gradient_descent_update(dcdw, lr, stop_gradients = True)
        >>> print(new_weights)
        ivy.array([[ 0.95,  1.98,  2.99],
        ...        [ 3.97,  5.94,  0.96],
        ...        [ 0.96, -0.07,  6.98]])

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
            Default is ``True``.
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
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, and also new mw and vw, following the adam
            updates.

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> w = ivy.array([1., 2, 3.])
        >>> dcdw = ivy.array([0.2,0.1,0.3])
        >>> lr = ivy.array(0.1)
        >>> vw_tm1 = ivy.zeros(1)
        >>> mw_tm1 = ivy.zeros(3)
        >>> step = 2
        >>> updated_weights = w.adam_update(dcdw, lr, mw_tm1, vw_tm1, step)
        >>> print(updated_weights)
        (ivy.array([0.92558753, 1.92558873, 2.92558718]),
        ivy.array([0.02, 0.01, 0.03]),
        ivy.array([4.00000063e-05, 1.00000016e-05, 9.00000086e-05]))

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
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the LAMB updates.

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> w = ivy.array([1., 2, 3])
        >>> dcdw = ivy.array([0.5,0.2,0.1])
        >>> lr = ivy.array(0.1)
        >>> vw_tm1 = ivy.zeros(1)
        >>> mw_tm1 = ivy.zeros(3)
        >>> step = ivy.array(1)
        >>> new_weights = w.lamb_update(dcdw, lr, mw_tm1, vw_tm1, step)
        >>> print(new_weights)
        (ivy.array([0.784, 1.78 , 2.78 ]),
        ivy.array([0.05, 0.02, 0.01]),
        ivy.array([2.5e-04, 4.0e-05, 1.0e-05]))

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
