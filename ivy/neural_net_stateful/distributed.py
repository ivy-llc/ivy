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
        self._dev_strs = dev_strs
        self._distributed_modules = list()
        with_v = True if 'v' in inspect.getfullargspec(module_class.__init__).args and ivy.exists(v) else False
        distributed_v = isinstance(v, ivy.Distributed)
        for i, dev_str in enumerate(dev_strs):
            if with_v:
                module = module_class(dev_str=dev_str, v=v[i] if distributed_v else v.to_dev(dev_str), **module_kwargs)
            else:
                module = module_class(dev_str=dev_str, **module_kwargs)
            self._distributed_modules.append(module)

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
        return ivy.Distributed(rets)

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

    @property
    def v(self):
        return ivy.Distributed([module.v for module in self._distributed_modules])
