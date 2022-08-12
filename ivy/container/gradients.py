from typing import Callable, Optional, Union, List, Dict

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
        exclusive: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_variable(
        self: ivy.Container,
        exclusive: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            self, exclusive, key_chains, to_apply, prune_unapplied, map_sequences
        )

    @staticmethod
    def static_execute_with_gradients(
        func: Callable,
        xs: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        retain_grads: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return ContainerBase.multi_map_in_static_method(
            "execute_with_gradients",
            func,
            xs,
            retain_grads=retain_grads,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_adam_step(
        dcdw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        mw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        vw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        step: Union[int, float],
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.adam_step. This method simply wraps
        the function, and so the docstring for ivy.adam_step also applies to this method
        with minimal changes.

        Parameters
        ----------
        dcdw
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
        >>> step = ivy.array([3.4])
        >>> beta1 = 0.87
        >>> beta2 = 0.976
        >>> epsilon = 1e-5
        >>> adam_step_delta = ivy.Container.static_adam_step(dcdw,\
                                mw, vw, step, beta1, beta2, epsilon)
        >>> print(adam_step_delta)
        {
            a: (list[3], <class ivy.array.Array> shape=[3]),
            b: (list[3], <class ivy.array.Array> shape=[3])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "adam_update",
            dcdw,
            mw,
            vw,
            step,
            beta1,
            beta2,
            epsilon,
        )

    def adam_step(
        self: ivy.Container,
        mw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
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
        return self.static_adam_step(self, mw, vw, step, beta1, beta2, epsilon)

    @staticmethod
    def static_optimizer_update(
        w,
        effective_grad,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "optimizer_update",
            w,
            effective_grad,
            lr,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def optimizer_update(
        self: ivy.Container,
        effective_grad,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_optimizer_update(
            self,
            effective_grad,
            lr,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_gradient_descent_update(
        w: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        dcdw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gradient_descent_update.
        This method simply wraps the function, and so the docstring for
        ivy.gradient_descent_update also applies to this method
        with minimal changes.

        Parameters
        ----------
        w
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
            The new weights, following the gradient descent updates.

        Examples
        --------
        With :code: `ivy.container` inputs:

        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([3.48, 5.72, 1.98]))
        >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),\
                                 b=ivy.array([2., 3.42, 1.69]))
        >>> lr = ivy.array(0.3)
        >>> NewWeights = ivy.Container.static_gradient_descent_update(w,\
                                                                    dcdw,\
                                                                    lr,\
                                                                    inplace=False)
        >>> print(NewWeights)
            {
                a: ivy.array([0.85, 1.94, 2.97]),
                b: ivy.array([2.88, 4.69, 1.47])
            }

        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([3.48, 5.72, 1.98]))
        >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),\
                                b=ivy.array([2., 3.42, 1.69]))
        >>> lr = ivy.Container(a=ivy.array(0.3),\
                                b=ivy.array(0.1))
        >>> NewWeights = ivy.Container.static_gradient_descent_update(w,\
                                                                      dcdw,\
                                                                      lr,\
                                                                      inplace=False)
        >>> print(NewWeights)
            {
                a: ivy.array([0.85, 1.94, 2.97]),
                b: ivy.array([3.28, 5.38, 1.81])
            }
        """
        return ContainerBase.multi_map_in_static_method(
            "gradient_descent_update",
            w,
            dcdw,
            lr,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def gradient_descent_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
        inplace
            Whether to perform the operation inplace, for backends which support inplace
            variable updates, and handle gradients behind the scenes such as PyTorch.
            If the update step should form part of a computation graph
            (i.e. higher order optimization), then this should be set to False.
            Default is True, provided the backend framework supports it.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is True.
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
            The new weights , following the gradient descent updates.

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
        return self.static_gradient_descent_update(
            self,
            dcdw,
            lr,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_lars_update(
        w,
        dcdw,
        lr,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "lars_update",
            w,
            dcdw,
            lr,
            decay_lambda=decay_lambda,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lars_update(
        self,
        dcdw,
        lr,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_lars_update(
            self,
            dcdw,
            lr,
            decay_lambda,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_adam_update(
        w,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "adam_update",
            w,
            dcdw,
            lr,
            mw_tm1=mw_tm1,
            vw_tm1=vw_tm1,
            step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adam_update(
        self,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_adam_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1,
            beta2,
            epsilon,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_lamb_update(
        w,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        max_trust_ratio=10,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "lamb_update",
            w,
            dcdw,
            lr,
            mw_tm1=mw_tm1,
            vw_tm1=vw_tm1,
            step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_trust_ratio=max_trust_ratio,
            decay_lambda=0,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lamb_update(
        self,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        max_trust_ratio=10,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_lamb_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1,
            beta2,
            epsilon,
            max_trust_ratio,
            decay_lambda,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_stop_gradient(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        preserve_type: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.stop_gradient. This method simply
        wraps the function, and so the docstring for ivy.stop_gradient also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Array or Container for which to stop the gradient.
        preserve_type
            Whether to preserve the input type (ivy.Variable or ivy.Array),
            otherwise an array is always returned. Default is True.
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
            preserve_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def stop_gradient(
        self: ivy.Container,
        preserve_type: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.stop_gradient. This method simply
        wraps the function, and so the docstring for ivy.stop_gradient also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Container for which to stop the gradient.
        preserve_type
            Whether to preserve the input type (ivy.Variable or ivy.Array),
            otherwise an array is always returned. Default is True.
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
            preserve_type,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
