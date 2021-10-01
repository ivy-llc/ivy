"""
Class for creating modules which are distributed over multiple devices
"""

# global
import os
import ivy
from typing import List


# Base #
# -----#

class DistributedModule:

    def __init__(self, module_class: ivy.Module, dev_strs: List[str] = None, v: ivy.Container = None,
                 build_mode: str = 'on_init', store_vars: bool = True):
        """
        Initialze Distributed Ivy Module, which is a stateful object consisting of trainable variables,
        distributed across arbitrarily many devices.

        :param module_class: The customized ivy.Module class you wish to distribute across devices.
        :type module_class: Customized ivy.Module class
        :param dev_strs: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
        :type dev_strs: str, optional
        :param v: Ivy container of trainable variables. Created internally by default.
        :type v: ivy container, optional
        :param build_mode: How the Module is built, either on initialization (now), explicitly by the user by calling
                           build(), or the first time the __call__ method is run. Default is on initialization.
        :type build_mode: str, optional
        :param store_vars: Whether or not to store the variables created. Default is True.
        :type store_vars: bool, optional
        :type build_mode: str, optional
        """
        self._build_mode = build_mode
        self._distributed_modules = list()
        self._dev_strs = dev_strs
        for dev_str in dev_strs:
            self._distributed_modules.append(module_class(dev_str, v.to_dev(dev_str), build_mode, store_vars))

    # Public #
    # -------#

    def __call__(self, *args, **kwargs):
        """
        the forward pass of the layer, treating layer instance as callable function.
        """
        args_dist, kwargs_dist = ivy.distribute(self._dev_strs, *args, **kwargs)
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
