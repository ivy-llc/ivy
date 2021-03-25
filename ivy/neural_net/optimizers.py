"""
Collection of Ivy optimizers.
"""

# global
import abc

# local
import ivy


# Base #
# -----#

class Optimizer(abc.ABC):

    def __init__(self, lr, dev_str, compile_step=False):
        """
        Construct an general Optimizer. This is an abstract class, and must be derived.

        :param lr: Learning rate.
        :type lr: float
        :param compile_step: Whether to compile the optimizer step, default is False.
        :type compile_step: bool, option
        """
        self._lr = lr
        self._compile_step = compile_step
        if compile_step:
            self._step_fn = ivy.compile_fn(self._step)
        else:
            self._step_fn = self._step

    # Public #
    # -------#

    # Abstract #

    @abc.abstractmethod
    def _step(self, v, grads):
        """
        Update nested variables container v from update step, using nested grads container.
        Override this abstract method with child class custom implementation.

        :param v: Nested variables to update.
        :type v: Ivy container of variables
        :param grads: Nested gradients to update.
        :type grads: sequence of arrays
        :return: The updated variables, following update step.
        """
        raise NotImplementedError

    # Given #

    def step(self, v, grads):
        """
        Update nested variables container v from possibly compiled overriden private self._step_fn

        :param v: Nested variables to update.
        :type v: Ivy container of variables
        :param grads: Nested gradients to update.
        :type grads: sequence of arrays
        :return: The updated variables, following update step.
        """
        return self._step_fn(v, grads)


# Optimizers #
# -----------#

class SGD(Optimizer):

    def __init__(self, lr=1e-4, compile_step=False):
        """
        Construct a Stochastic-Gradient-Descent (SGD) optimizer.

        :param lr: Learning rate, default is 1e-4.
        :type lr: float, optional
        :param compile_step: Whether to compile the optimizer step, default is False.
        :type compile_step: bool, option
        """
        Optimizer.__init__(self, lr, compile_step)

    # Custom Step

    def _step(self, v, grads):
        """
        Update nested variables container v by gradient descent step, using nested gradients container.

        :param v: Nested variables to update.
        :type v: Ivy container of variables
        :param grads: Nested gradients to update.
        :type grads: sequence of arrays
        :return: The new updated variables container, following gradient descent step.
        """
        return ivy.gradient_descent_update(v, grads, self._lr)


class Adam(Optimizer):

    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-07, compile_step=False, dev_str='cpu'):
        """
        Construct an ADAM optimizer.

        :param lr: Learning rate, default is 1e-4.
        :type lr: float, optional
        :param beta1: gradient forgetting factor, default is 0.9
        :type beta1: float, optional
        :param beta2: second moment of gradient forgetting factor, default is 0.999
        :type beta2: float, optional
        :param epsilon: divisor during adam update, preventing division by zero, default is 1e-07
        :type epsilon: float, optional
        :param compile_step: Whether to compile the optimizer step, default is False.
        :type compile_step: bool, option
        :param dev_str: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc. Default is cpu.
        :type dev_str: str, optional
        """
        Optimizer.__init__(self, lr, compile_step)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._mw = None
        self._vw = None
        self._first_pass = True
        self._step = ivy.array([0], dev_str=dev_str)

    # Custom Step

    def _step(self, v, grads):
        """
        Update nested variables container v by Adam update step, using nested grads container.

        :param v: Nested variables to update.
        :type v: Ivy container of variables
        :param grads: Nested gradients to update.
        :type grads: sequence of arrays
        :return: The updated variables, following Adam update step.
        """
        if self._first_pass:
            self._mw = grads
            self._vw = grads.map(lambda x, _: x ** 2)
            self._first_pass = False
        new_v, self._mw, self._vw = ivy.adam_update(
            v, grads, self._lr, self._mw, self._vw, self._step, self._beta1, self._beta2, self._epsilon)
        self._step += 1
        return new_v
