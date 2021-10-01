"""
Class for creating modules which are distributed over multiple devices
"""

# global
import os
import ivy
import inspect


# Module #
# -------#

# noinspection PyArgumentList
class DistributedModule:

    def __init__(self, module_class, dev_strs, v=None, **module_kwargs):
        """
        Initialze Distributed Ivy Module, which is a stateful object consisting of trainable variables,
        distributed across arbitrarily many devices.

        :param module_class: The customized ivy.Module class you wish to distribute across devices.
        :type module_class: Customized ivy.Module class
        :param dev_strs: devices on which to distribute the module.
        :type dev_strs: str
        :param v: Ivy container of trainable variables. Created internally by default.
        :type v: ivy container, optional
        :param build_mode: How the Module is built, either on initialization (now), explicitly by the user by calling
                           build(), or the first time the __call__ method is run. Default is on initialization.
        :type build_mode: str, optional
        :param store_vars: Whether or not to store the variables created. Default is True.
        :type store_vars: bool, optional
        :param module_kwargs: Dict of keywords arguments to pass the module constructor. Default is None.
        :type module_kwargs: dict of any, optional
        :type build_mode: str, optional
        """
        self._distributed_modules = list()
        self._dev_strs = dev_strs
        if 'v' in inspect.getfullargspec(module_class.__init__).args:
            module_kwargs['v'] = ivy.default(lambda: v.to_dev(dev_str), None, True)
        for dev_str in dev_strs:
            self._distributed_modules.append(module_class(
                dev_str=dev_str, **module_kwargs))

    # Public #
    # -------#

    def __call__(self, *args, **kwargs):
        """
        the forward pass of the layer, treating layer instance as callable function.
        """
        args_dist, kwargs_dist = ivy.distribute_args(self._dev_strs, *args, **kwargs)
        rets = list()
        for module, args_d, kwargs_d in zip(self._distributed_modules, args_dist, kwargs_dist):
            rets.append(module(*args_d, **kwargs_d))
        return rets

    def save_weights(self, weights_path):
        """
        Save the weights on the Module.
        :param weights_path: The hdf5 file for saving the weights.
        :type weights_path: string
        """
        all_vs = [module.v for module in self._distributed_modules]
        assert ivy.Container.identical_structure(all_vs)
        assert ivy.Container.multi_map(lambda xs, _: ivy.arrays_equal(xs), all_vs).all_true()
        v = all_vs[0]
        os.makedirs('/'.join(weights_path.split('/')[:-1]), exist_ok=True)
        v.to_disk_as_hdf5(weights_path)

    def build(self, *args, from_call=False, **kwargs):
        """
        Build the internal layers and variables for each of the distributed modules.
        """
        [module.build(*args, from_call, **kwargs) for module in self._distributed_modules]

    # Properties #
    # -----------#

    @property
    def build_mode(self):
        return self._build_mode

    @property
    def built(self):
        return min([module.built for module in self._distributed_modules])


# Optimizer #
# ----------#

# noinspection PyCallingNonCallable
class DistributedOptimizer:

    def __init__(self, optimizer_class, dev_strs, lr, compile_step=False, inplace=True, stop_gradients=True,
                 **optimizer_kwargs):
        """
        Construct an general Optimizer. This is an abstract class, and must be derived.

        :param optimizer_class: The customized ivy.Optimizer class you wish to distribute across devices.
        :type optimizer_class: Customized ivy.Optimizer class
        :param dev_strs: devices on which to distribute the module.
        :type dev_strs: str
        :param lr: Learning rate.
        :type lr: function or float.
        :param compile_step: Whether to compile the optimizer step, default is False.
        :type compile_step: bool, optional
        :param inplace: Whether to update the variables in-place, or to create new variable handles.
                        This is only relevant for frameworks with stateful variables such as PyTorch. Default is True.
        :type inplace: bool, optional
        :param stop_gradients: Whether to stop the gradients of the variables after each gradient step. Default is True.
        :type stop_gradients: bool, optional
        :param optimizer_kwargs: Dict of keywords arguments to pass the optimizer constructor. Default is None.
        :type optimizer_kwargs: dict of any, optional
        """
        self._distributed_optimizers = list()
        self._dev_strs = dev_strs
        for dev_str in dev_strs:
            self._distributed_optimizers.append(optimizer_class(
                lr=lr, compile_step=compile_step, inplace=inplace, stop_gradients=stop_gradients, dev_str=dev_str,
                **optimizer_kwargs))

    def set_state(self, state):
        """
        Set state of the optimizer.

        :param state: Nested state to update.
        :type state: Ivy container of state tensors
        """
        [optim.set_state(s_sub) for optim, s_sub in zip(self._distributed_optimizers, state)]

    def step(self, v, grads, ignore_missing=False):
        """
        Update nested variables container v from possibly compiled overriden private self._step_fn

        :param v: Nested variables to update.
        :type v: Ivy container of variables
        :param grads: Nested gradients to update.
        :type grads: sequence of arrays
        :param ignore_missing: Whether to ignore keys missing from the gradients which exist in the variables.
                               Default is False.
        :type ignore_missing: bool, optional
        :return: The updated variables, following update step.
        """
        [optim.step(v_sub, g_sub, ignore_missing)
         for optim, v_sub, g_sub in zip(self._distributed_optimizers, v, grads)]
