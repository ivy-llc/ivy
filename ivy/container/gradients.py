from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithGradients(ContainerBase):
    @staticmethod
    def static_variable(
        x: ivy.Container,
        key_chains: Union[bool, ivy.Container] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.variable. This method simply wraps
        the function, and so the docstring for ivy.variable also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            An ivy container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        ret
            A container with ivy variables, that supports gradient computation.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> ivy.set_backend('torch')
        >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([3., 4.]))
        >>> y = ivy.Container.static_variable(x)
        >>> y
        {
            a: ivy.array([1., 2.]),
            b: ivy.array([3., 4.])
        }
        >>> ivy.unset_backend()
        """
        return ContainerBase.multi_map_in_static_method(
            "variable",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def variable(
        self: ivy.Container,
        key_chains: Union[bool, ivy.Container] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.variable. This method simply
        wraps the function, and so the docstring for ivy.variable also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            An ivy container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        ret
            A container with ivy variables, that supports gradient computation.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> ivy.set_backend('jax')
        >>> x = ivy.Container(a=ivy.array([0.3, 1.]), b=ivy.array([-1., 2.2]))
        >>> y = x.variable()
        >>> y
        {
            a: ivy.array([0.3, 1.]),
            b: ivy.array([-1., 2.2])
        }
        >>> ivy.unset_backend()
        """
        return self.static_variable(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_variable(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        exclusive: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.is_variable. This method simply wraps
        the function, and so the docstring for ivy.is_variable also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            An ivy container.
        exclusive
            Whether to check if the data type is exclusively a variable, rather than an
            array. For frameworks like JAX that do not have exclusive variable types,
            the function will always return False if this flag is set, otherwise the
            check is the same for general arrays. Default is False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        ret
            Boolean, true if x is a trainable variable, false otherwise.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> is_var = ivy.Container.static_is_variable(x)
        >>> print(is_var)
        {
            a: false,
            b: false
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=False, b=True)
        >>> is_var = ivy.Container.static_is_variable(x, exclusive=exclusive)
        >>> print(is_var)
        {
            a: false,
            b: false
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "is_variable",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            exclusive=exclusive,
        )

    def is_variable(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        exclusive: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.is_variable. This method simply
        wraps the function, and so the docstring for ivy.is_variable also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            An ivy container.
        exclusive
            Whether to check if the data type is exclusively a variable, rather than an
            array. For frameworks like JAX that do not have exclusive variable types,
            the function will always return False if this flag is set, otherwise the
            check is the same for general arrays. Default is False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        ret
            Boolean, true if x is a trainable variable, false otherwise.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> is_var = x.is_variable(exclusive=True)
        >>> print(is_var)
        {
            a: false,
            b: false
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=True, b=True)
        >>> is_var = x.is_variable(exclusive=exclusive)
        >>> print(is_var)
        {
            a: false,
            b: false
        }
        """
        return self.static_is_variable(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            exclusive=exclusive,
        )

    @staticmethod
    def static_variable_data(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.variable_data. This method simply
        wraps the function, and so the docstring for ivy.variable_data also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            An ivy variable or container of variables.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        ret
            The internal data stored by the variable.

        """
        return ContainerBase.multi_map_in_static_method(
            "variable_data",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def variable_data(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.variable_data. This method simply
        wraps the function, and so the docstring for ivy.variable_data also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            An ivy container of variables.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        ret
            The internal data stored by the variable.

        """
        return self.static_variable_data(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adam_step(
        self: ivy.Container,
        mw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.adam_step. This method simply wraps
        the function, and so the docstring for ivy.adam_step also applies to this method
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
        with :code: `ivy.container` inputs:

        >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),\
                         b=ivy.array([3., 4., 5.]))
        >>> mw = ivy.Container(a=ivy.array([0., 0., 0.]),\
                               b=ivy.array([0., 0., 0.]))
        >>> vw = ivy.Container(a=ivy.array([0.,]),\
                               b=ivy.array([0.,]))
        >>> step = ivy.array(1)
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step, beta1, beta2, epsilon)
        {
            a: (list[3], <class ivy.array.Array> shape=[3]),
            b: (list[3], <class ivy.array.Array> shape=[3])
        }
        """
        return ivy.adam_step(
            self,
            mw,
            vw,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

    def optimizer_update(
        self: ivy.Container,
        effective_grad: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        stop_gradients: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Update weights ws of some function, given the true or effective derivatives
        of some cost c with respect to ws, [dc/dw for w in ws].

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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the optimizer updates.

        """
        return ivy.optimizer_update(
            self, effective_grad, lr, stop_gradients=stop_gradients, out=out
        )

    def gradient_descent_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        stop_gradients: bool = True,
        out: ivy.Container = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gradient_descent_update.
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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The new weights, following the gradient descent updates.

        Examples
        --------
        With :code: `ivy.container` inputs:

        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([3.48, 5.72, 1.98]))
        >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),\
                                 b=ivy.array([2., 3.42, 1.69]))
        >>> lr = ivy.array(0.3)
        >>> NewWeights = w.gradient_descent_update(dcdw, lr, inplace=False)
        >>> print(NewWeights)
            {
                a: ivy.array([0.85, 1.94, 2.97]),
                b: ivy.array([2.88,4.69,1.47])
            }

        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([3.48, 5.72, 1.98]))
        >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),\
                                 b=ivy.array([2., 3.42, 1.69]))
        >>> lr = ivy.Container(a=ivy.array(0.3),\
                                b=ivy.array(0.1))
        >>> NewWeights = w.gradient_descent_update(dcdw, lr, inplace=False)
        >>> print(NewWeights)
            {
                a: ivy.array([0.85, 1.94, 2.97]),
                b: ivy.array([3.28, 5.38, 1.81])
            }
        """
        return ivy.gradient_descent_update(
            self,
            dcdw,
            lr,
            stop_gradients=stop_gradients,
            out=out,
        )

    def lars_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        decay_lambda: float = 0,
        stop_gradients: bool = True,
        out: Optional[ivy.Container] = None,
    ):
        """Update weights ws of some function, given the derivatives of some cost c with
        respect to ws, [dc/dw for w in ws], by applying Layerwise Adaptive Rate Scaling
        (LARS) method.

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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

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
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        mw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: int,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        stop_gradients: bool = True,
        out: Optional[ivy.Container] = None,
    ):
        """Update weights ws of some function, given the derivatives of some cost c with
        respect to ws, using ADAM update. `[reference]

        <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

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
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        mw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: int,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        max_trust_ratio: Union[int, float] = 10,
        decay_lambda: float = 0,
        stop_gradients: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Update weights ws of some function, given the derivatives of some cost c with
        respect to ws, [dc/dw for w in ws], by applying LAMB method.

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
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

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

    @staticmethod
    def static_stop_gradient(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        preserve_type: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.stop_gradient. This method simply
        wraps the function, and so the docstring for ivy.stop_gradient also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Array or Container for which to stop the gradient.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        preserve_type
            Whether to preserve the input type (ivy.Variable or ivy.Array),
            otherwise an array is always returned. Default is True.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The same array x, but with no gradient information.
        """
        return ContainerBase.multi_map_in_static_method(
            "stop_gradient",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            preserve_type=preserve_type,
            out=out,
        )

    def stop_gradient(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        preserve_type: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.stop_gradient. This method simply
        wraps the function, and so the docstring for ivy.stop_gradient also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Container for which to stop the gradient.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        preserve_type
            Whether to preserve the input type (ivy.Variable or ivy.Array),
            otherwise an array is always returned. Default is True.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The same array x, but with no gradient information.
        """
        return self.static_stop_gradient(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            preserve_type=preserve_type,
            out=out,
        )
