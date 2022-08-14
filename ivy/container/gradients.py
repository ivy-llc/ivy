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

        >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([3., 4.]))
        >>> y = ivy.Container.static_variable(x)
        >>> y
        {
            a: ivy.array([1., 2.]),
            b: ivy.array([3., 4.])
        }

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

        >>> x = ivy.Container(a=ivy.array([0.3, 1.]), b=ivy.array([-1., 2.2]))
        >>> y = x.variable()
        >>> y
        {
            a: ivy.array([0.3, 1.]),
            b: ivy.array([-1., 2.2])
        }

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

        >>> x = ivy.Container(a = ivy.array(3.2), b=ivy.array(2))
        >>> is_var = ivy.Container.static_is_variable(x, exclusive=True)
        >>> print(is_var)
        {
            a: false,
            b: false
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.variable(ivy.array([2.0, -1.0, 0.0])),\
                              b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=False, b=True)
        >>> is_var = ivy.Container.static_is_variable(x, exclusive=exclusive)
        >>> print(is_var)
        {
            a: true,
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

    static_is_variable.computes_gradients = True


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
        >>> x = ivy.Container(a = ivy.array(3.2), b=ivy.array(2))
        >>> is_var = ivy.Container.static_is_variable(x, exclusive=True)
        >>> print(is_var)
        {
            a: false,
            b: false
        }
        With multiple :code:`ivy.Container` inputs:
        >>> x = ivy.Container(a=ivy.variable(ivy.array([2.0, -1.0, 0.0])),\
                              b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=False, b=True)
        >>> is_var = ivy.Container.static_is_variable(x, exclusive=exclusive)
        >>> print(is_var)
        {
            a: true,
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

    static_is_variable.computes_gradients = True

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
        >>> x = ivy.Container(a = ivy.array(3.2), b=ivy.array(2))
        >>> is_var = x.is_variable(exclusive=True)
        >>> print(is_var)
        {
            a: false,
            b: false
        }
        With multiple :code:`ivy.Container` inputs:
        >>> x = ivy.Container(a=ivy.variable(ivy.array([2.0, -1.0, 0.0])),\
                              b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=False, b=True)
        >>> is_var = x.is_variable(exclusive=exclusive)
        >>> print(is_var)
        {
            a: true,
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

    is_variable.computes_gradients = True

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
        """
        ivy.Container static method variant of ivy.execute_with_gradients.
        This method simply wraps the function, and so the docstring for 
        ivy.execute_with_gradients also applies to this method with minimal changes.
        
        Call function func with input of xs variables, and return func first output y,
        the gradients [dy/dx for x in xs], and any other function outputs after 
        the returned y value.

        Parameters
        ----------
        func
            Function for which we compute the gradients of the output with respect to xs
            input.
        xs
            Variables for which to compute the function gradients with respective to.
        retain_grads
            Whether to retain the gradients of the returned values.
            (Default value = False)

        Returns
        -------
        ret
            the function first output y, the gradients [dy/dx for x in xs], 
            and any other extra function outputs.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> ivy.set_backend('tensorflow')
        >>> func = lambda x : 2*x**2
        >>> xs = ivy.Container(a = ivy.array([1.,1.,1.]))
        >>> results = ivy.Container.static_execute_with_gradients(func, \
            xs)
        >>> print(results)
        {a: (list[2], <class ivy.array.array.Array> shape=[3])}
        
        With multiple :code:`ivy.Container` inputs:

        >>> func = lambda x: x**2
        >>> xs = ivy.Container(a=ivy.array([1.,1.,1.]), \
            b =ivy.array([5.,5.,5.]))
        >>> results = ivy.Container.static_execute_with_gradients(func, \
                    xs)
        >>> print(results)
        {
        a: (list[2], <class ivy.array.array.Array> shape=[3]),
        b: (list[2], <class ivy.array.array.Array> shape=[3])
        }
        """
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

    def execute_with_gradients(
        self: ivy.Container,
        func: Callable,
        retain_grads: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        """
        ivy.Container instance method variant of ivy.execute_with_gradients.
        This method simply wraps the function, and so the docstring for 
        ivy.execute_with_gradients also applies to this method with minimal changes.
        
        Call function func with the container, and return func first output y,
        the gradients [dy/dx for x in xs], and any other function outputs after 
        the returned y value.

        Parameters
        ----------
        func
            Function for which we compute the gradients of the output with respect to xs
            input.
        retain_grads
            Whether to retain the gradients of the returned values.
            (Default value = False)

        Returns
        -------
        ret
            the function first output y, the gradients [dy/dx for x in xs], 
            and any other extra function outputs.

        Examples
        --------

        With :code:`ivy.Container` input:

        >>> ivy.set_backend('tensorflow')
        >>> func = lambda x : 2*x**2
        >>> xs = ivy.Container(a = ivy.array([1.,1.,1.]))
        >>> results = xs.execute_with_gradients(func)
        >>> print(results)
        {a: (list[2], <class ivy.array.array.Array> shape=[3])}

        With multiple :code:`ivy.Container` inputs:

        >>> func = lambda x: x**2
        >>> xs = ivy.Container(a=ivy.array([1.,1.,1.]), \
            b =ivy.array([5.,5.,5.]))
        >>> results = xs.execute_with_gradients(func)
        >>> print(results)
        {a: (list[2], <class ivy.array.array.Array> shape=[3]),
        b: (list[2], <class ivy.array.array.Array> shape=[3])}
        """
        return self.static_execute_with_gradients(
            func,
            self,
            retain_grads=retain_grads,
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
        out: Optional[ivy.Container] = None,
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.
        Returns
        -------
        ret
            The adam step delta.
        Examples
        --------
        With one :code:`ivy.Container` input:
        >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),\
                                 b=ivy.array([3., 4., 5.]))
        >>> mw = ivy.array([1., 4., 9.])
        >>> vw = ivy.array([0.,])
        >>> step = ivy.array([3.4])
        >>> beta1 = 0.87
        >>> beta2 = 0.976
        >>> epsilon = 1e-5
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step, beta1=beta1, beta2=beta2,\
                                             epsilon=epsilon)
        >>> print(adam_step_delta)
        ({
            a: ivy.array([6.49e+04, 1.74e+01, 1.95e+01]),
            b: ivy.array([2.02, 4.82, 8.17])
        }, {
            a: ivy.array([0.87, 3.61, 8.09]),
            b: ivy.array([1.26, 4., 8.48])
        }, {
            a: ivy.array([0., 0.024, 0.096]),
            b: ivy.array([0.216, 0.384, 0.6])
        })
        
        With multiple :code:`ivy.Container` inputs:
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
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step, beta1=beta1, beta2=beta2,\
                                             epsilon=epsilon)
        >>> print(adam_step_delta)
        ({
            a: ivy.array([0., 0.626, 0.626]),
            b: ivy.array([0.626, 0.626, 0.626])
        }, {
            a: ivy.array([0., 0.13, 0.26]),
            b: ivy.array([0.39, 0.52, 0.65])
        }, {
            a: ivy.array([0., 0.024, 0.096]),
            b: ivy.array([0.216, 0.384, 0.6])
        })
        """
        return ivy.adam_step(
            self, mw, vw, step, beta1=beta1, beta2=beta2, epsilon=epsilon, out=out
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
        Examples
        --------
        With one :code:`ivy.Container` input:
        
        >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),\
                            b=ivy.array([3., 4., 5.]))
        >>> effective_grad = ivy.array([0., 0., 0.])
        >>> lr = 3e-4
        >>> ws_new = w.optimizer_update(effective_grad, lr)
        >>> print(ws_new)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([3., 4., 5.])
        }
        With multiple :code:`ivy.Container` inputs:
        
        >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),\
                              b=ivy.array([3., 4., 5.]))
        >>> effective_grad = ivy.Container(a=ivy.array([0., 0., 0.]),\
                                           b=ivy.array([0., 0., 0.]))
        >>> lr = 3e-4
        >>> ws_new = w.optimizer_update(effective_grad, lr, out=w)
        >>> print(w)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([3., 4., 5.])
        }
        
        >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),\
                            b=ivy.array([3., 4., 5.]))
        >>> effective_grad = ivy.Container(a=ivy.array([0., 0., 0.]),\
                                        b=ivy.array([0., 0., 0.]))
        >>> lr = ivy.array([3e-4])
        >>> ws_new = w.optimizer_update(effective_grad, lr, stop_gradients=False)
        >>> print(ws_new)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([3., 4., 5.])
        }
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
        With one :code:`ivy.Container` inputs:
        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([3.48, 5.72, 1.98]))
        >>> dcdw = ivy.array([0.5, 0.2, 0.1])
        >>> lr = ivy.array(0.3)
        >>> w_new = w.gradient_descent_update(dcdw, lr)
        >>> print(w_new)
        {
            a: ivy.array([0.85, 1.94, 2.97]),
            b: ivy.array([3.33, 5.66, 1.95])
        }
        
        With multiple :code:`ivy.Container` inputs:
        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([3.48, 5.72, 1.98]))
        >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),\
                                 b=ivy.array([2., 3.42, 1.69]))
        >>> lr = ivy.array(0.3)
        >>> w_new = w.gradient_descent_update(dcdw, lr)
        >>> print(w_new)
        {
            a: ivy.array([0.85, 1.94, 2.97]),
            b: ivy.array([2.88, 4.69, 1.47])
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
        Examples
        --------
        With one :code:`ivy.Container` inputs:
        >>> w = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([4., 5., 6.]))
        >>> dcdw = ivy.array([3., 4., 5.])
        >>> mw_tm1 = ivy.array([0., 0., 0.])
        >>> vw_tm1 = ivy.array([0.])
        >>> lr = ivy.array(1.)
        >>> step = ivy.array([2])
        >>> new_weights = w.lamb_update(dcdw, mw_tm1, vw_tm1, lr, step)
        >>> print(new_weights)
        ({
            a: ivy.array([1., 2., 3.]),
            b: ivy.array([4., 5., 6.])
        }, ivy.array([0.3, 0.4, 0.5]), ivy.array([1.01, 1.01, 1.02]))
        With multiple :code:`ivy.Container` inputs:
        
        >>> w = ivy.Container(a=ivy.array([1.,3.,5.]),\
                              b=ivy.array([3.,4.,2.]))
        >>> dcdw = ivy.Container(a=ivy.array([0.2,0.3,0.6]),\
                                 b=ivy.array([0.6,0.4,0.7]))
        >>> mw_tm1 = ivy.Container(a=ivy.array([0.,0.,0.]),\
                                   b=ivy.array([0.,0.,0.]))
        >>> vw_tm1 = ivy.Container(a=ivy.array([0.,]),\
                                   b=ivy.array([0.,]))
        >>> step = ivy.array([3.4])
        >>> beta1 = 0.9
        >>> beta2 = 0.999
        >>> epsilon = 1e-7
        >>> max_trust_ratio = 10
        >>> decay_lambda = 0
        >>> stop_gradients = True
        >>> lr = ivy.array(0.5)
        >>> new_weights = w.lamb_update(dcdw, lr, mw_tm1, vw_tm1, step, beta1=beta1,\
                                        beta2=beta2, epsilon=epsilon,\
                                        max_trust_ratio=max_trust_ratio,\
                                        decay_lambda=decay_lambda,\
                                        stop_gradients=stop_gradients)
        >>> print(new_weights)
        ({
            a: ivy.array([-0.708, 1.29, 3.29]),
            b: ivy.array([1.45, 2.45, 0.445])
        }, {
            a: ivy.array([0.02, 0.03, 0.06]),
            b: ivy.array([0.06, 0.04, 0.07])
        }, {
            a: ivy.array([4.0e-05, 9.0e-05, 3.6e-04]),
            b: ivy.array([0.00036, 0.00016, 0.00049])
        })
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