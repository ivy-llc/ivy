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

    def __init__(self, lr, compile_step=False):
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
    def _step(self, *args, **kwargs):
        """
        The custom step function for the optimizer, must be implemented in child class.
        """
        raise NotImplementedError

    # Given #

    def step(self, *args, **kwargs):
        """
        The callable step function, which calls the private step function, either compiled or not compiled.
        """
        return self._step_fn(*args, **kwargs)


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
