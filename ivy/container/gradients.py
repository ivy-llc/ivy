from typing import Callable, Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithGradients(ContainerBase):
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
        """
        ivy.Container static method variant of ivy.execute_with_gradients. This method simply wraps
        the function, and so the docstring for ivy.execute_with_gradients also applies to this
        method with minimal changes.
        
        Call function func with input of xs variables, and return func first output y,
        the gradients [dy/dx for x in xs], and any other function outputs after the returned
        y value.

        Parameters
        ----------
        func
            Function for which we compute the gradients of the output with respect to xs
            input.
        xs
            Variables for which to compute the function gradients with respective to.
        retain_grads
            Whether to retain the gradients of the returned values. (Default value = False)

        Returns
        -------
        ret
            the function first output y, the gradients [dy/dx for x in xs], and any other
            extra function outputs.

        Examples
        --------

        With :code:`ivy.Container` input:

        >>> ivy.set_backend('tensorflow')
        >>> z  = ivy.variable(ivy.array([2.,1.,100.]))
        >>> func = lambda x :ivy.matmul(z,x)
        >>> xs = ivy.Container(a = ivy.array([1.,1.,1.]))
        >>> results = ivy.Container.static_execute_with_gradients(
        >>>            func,
        >>>            xs)
        >>> func_output,grads = results['a']
        >>> print("function output: ", func_output)
        >>> print("grads: ", grads)
        function output:  ivy.array(103.)
        grads:  ivy.array([  2.,   1., 100.])
        
        With multiple :code:`ivy.Container` inputs:

        >>> func = lambda x: x**2
        >>> xs = ivy.Container(
        >>>            a=ivy.array([1.,1.,1.]),
        >>>            b =ivy.array([5.,5.,5.]) )
        >>> results = ivy.Container.static_execute_with_gradients(
        >>>            func,
        >>>            xs)
        >>> a_func_output, a_grads = results['a']
        >>> b_func_output, b_grads = results['b']
        >>> print("a function output: ", a_func_output)
        >>> print("a gradients: ", a_grads)
        >>> print("b function output: ", b_func_output)
        >>> print("b gradients: ", b_grads)
        a function output:  ivy.array([1., 1., 1.])
        a gradients:  ivy.array([2., 2., 2.])
        b function output:  ivy.array([25., 25., 25.])
        b gradients:  ivy.array([10., 10., 10.])

        >>> linear = ivy.Linear(3,1)
        >>> func = lambda x: linear(x)
        >>> xs = ivy.Container(
        >>>            a=ivy.array([1.,1.,1.]),
        >>>            b =ivy.array([5.,5.,5.]),
        >>>            c=ivy.array([1.,0.,1.]) )
        >>> results = ivy.Container.static_execute_with_gradients(
        >>>            func,
        >>>            xs)
        >>> a_func_output, a_grads = results['a']
        >>> b_func_output, b_grads = results['b']
        >>> c_func_output, c_grads = results['c']
        >>> print("a function output: ", a_func_output)
        >>> print("a gradients: ", a_grads)
        >>> print("b function output: ", b_func_output)
        >>> print("b gradients: ", b_grads)
        >>> print("c function output: ", c_func_output)
        >>> print("c gradients: ", c_grads)
        a function output:  ivy.array([-0.104])
        a gradients:  ivy.array([ 0.138,  0.971, -1.21 ])
        b function output:  ivy.array([-0.521])
        b gradients:  ivy.array([ 0.138,  0.971, -1.21 ])
        c function output:  ivy.array([-1.08])
        c gradients:  ivy.array([ 0.138,  0.971, -1.21 ])
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
        ivy.Container instance method variant of ivy.execute_with_gradients. This method simply wraps
        the function, and so the docstring for ivy.execute_with_gradients also applies to this
        method with minimal changes.
        
        Call function func with the container, and return func first output y,
        the gradients [dy/dx for x in xs], and any other function outputs after the returned
        y value.

        Parameters
        ----------
        func
            Function for which we compute the gradients of the output with respect to xs
            input.
        retain_grads
            Whether to retain the gradients of the returned values. (Default value = False)

        Returns
        -------
        ret
            the function first output y, the gradients [dy/dx for x in xs], and any other
            extra function outputs.

        Examples
        --------

        With :code:`ivy.Container` input:

        >>> ivy.set_backend('tensorflow')
        >>> z  = ivy.variable(ivy.array([2.,1.,100.]))
        >>> func = lambda x :ivy.matmul(z,x)
        >>> xs = ivy.Container(a = ivy.array([1.,1.,1.]))
        >>> results = xs.execute_with_gradients(
        >>>            func)
        >>> func_output,grads = results['a']
        >>> print("function output: ", func_output)
        >>> print("grads: ", grads)
        function output:  ivy.array(103.)
        grads:  ivy.array([  2.,   1., 100.])

        With multiple :code:`ivy.Container` inputs:

        >>> func = lambda x: x**2
        >>> xs = ivy.Container(
        >>>            a=ivy.array([1.,1.,1.]),
        >>>            b =ivy.array([5.,5.,5.]) )
        >>> results = xs.execute_with_gradients(
        >>>            func)
        >>> a_func_output, a_grads = results['a']
        >>> b_func_output, b_grads = results['b']
        >>> print("a function output: ", a_func_output)
        >>> print("a gradients: ", a_grads)
        >>> print("b function output: ", b_func_output)
        >>> print("b gradients: ", b_grads)
        a function output:  ivy.array([1., 1., 1.])
        a gradients:  ivy.array([2., 2., 2.])
        b function output:  ivy.array([25., 25., 25.])
        b gradients:  ivy.array([10., 10., 10.])


        >>> linear = ivy.Linear(3,1)
        >>> func = lambda x: linear(x)
        >>> xs = ivy.Container(
        >>>    a=ivy.array([1.,1.,1.]),
        >>>    b =ivy.array([5.,5.,5.]),
        >>>    c=ivy.array([1.,0.,1.]) )
        >>> results = xs.execute_with_gradients(
        >>>    func)
        >>> a_func_output, a_grads = results['a']
        >>> b_func_output, b_grads = results['b']
        >>> c_func_output, c_grads = results['c']
        >>> print("a function output: ", a_func_output)
        >>> print("a gradients: ", a_grads)
        >>> print("b function output: ", b_func_output)
        >>> print("b gradients: ", b_grads)
        >>> print("c function output: ", c_func_output)
        >>> print("c gradients: ", c_grads)
        a function output:  ivy.array([2.36])
        a gradients:  ivy.array([1.15  , 1.12  , 0.0972])
        b function output:  ivy.array([11.8])
        b gradients:  ivy.array([1.15  , 1.12  , 0.0972])
        c function output:  ivy.array([1.24])
        c gradients:  ivy.array([1.15  , 1.12  , 0.0972])
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
        w,
        dcdw,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:

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
        self,
        dcdw,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
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
