# global
import abc
from typing import Union, Callable, Optional
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

    def gradient_descent_update(
        self: ivy.Array,
        dcdw: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        inplace=None,
        stop_gradients=True,
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
        return ivy.gradient_descent_update(self, dcdw, lr, inplace, stop_gradients)
        
    def execute_with_gradients(
        self: ivy.Array,
        func: Callable,
        retain_grads: bool = False
    ):
        """
        ivy.Array instance method variant of ivy.execute_with_gradients.
        This method simply wraps the function, and so 
        the docstring for ivy.execute_with_gradients also applies to this
        method with minimal changes.
        
        Call function func with the container, and return func first output y,
        the gradients [dy/dx for x in xs], and any other function 
        outputs after the returned y value.

        Parameters
        ----------
        func
            Function for which we compute the gradients of the 
            output with respect to xs input.
        retain_grads
            Whether to retain the gradients of the returned values.
            (Default value = False)
            the function first output y, the gradients [dy/dx for x in xs],
            and any other extra function outputs.

        Examples
        --------

        With :code:`ivy.Array` input:

        >>> ivy.set_backend('tensorflow')
        >>> func = lambda x :x**2
        >>> xs =ivy.array([1.,0.,10.])
        >>> func_output,grads = xs.execute_with_gradients(func)
        >>> print("function output: ", func_output)
        >>> print("grads: ", grads)
        function output:  ivy.array([  1.,   0., 100.])
        grads:  ivy.array([ 2.,  0., 20.])

        >>> func = lambda x :2*x**2
        >>> xs = ivy.array([1.,1.,1.])
        >>> func_output, grads  = xs.execute_with_gradients(func)
        >>> print("function output: ", func_output)
        >>> print("grads: ", grads)
        function output:  ivy.array([2., 2., 2.])
        grads:  ivy.array([4., 4., 4.])    
        >>> ivy.unset_backend()
        """
        return ivy.execute_with_gradients(
            func,
            self,
            retain_grads=retain_grads
        )
        
    def stop_gradient(
        self: ivy.Array, preserve_type: bool = True, *, out: Optional[ivy.Array] = None
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
        return ivy.stop_gradient(self._data, preserve_type, out=out)
