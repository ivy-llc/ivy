"""
Base class for deriving trainable modules
"""

# global
import os
import abc

# local
import ivy
from ivy.core.container import Container


# Base #
# -----#

class Module(abc.ABC):

    def __init__(self, dev_str=None, v=None, build_mode='on_init'):
        """
        Initialze Ivy layer, which is a stateful object consisting of trainable variables.

        :param dev_str: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
        :type dev_str: str, optional
        :param v: Ivy container of trainable variables. Created internally by default.
        :type v: ivy container, optional
        :param build_mode: How the Module is built, either on initialization (now), explicitly by the user by calling
                           build(), or the first time the __call__ method is run. Default is on initialization.
        :type build_mode: str, optional
        """
        valid_build_modes = ['on_init', 'explicit', 'on_call']
        if build_mode not in valid_build_modes:
            raise Exception('build_mode must be one of {} of type str, but found {} of type{}'.format(
                valid_build_modes, build_mode, type(build_mode)))
        if dev_str is None:
            dev_str = 'gpu:0' if ivy.gpu_is_available() else 'cpu'
        self._dev_str = dev_str
        self._deffered_build = build_mode == 'on_call'
        self._explicit_build = build_mode == 'explicit'
        self._built = False
        self.v = v
        if build_mode != 'on_init':
            return
        self.build()

    # Private #
    # --------#

    def _fn_with_var_arg(self, fn, v_fn):
        def new_fn(*a, with_grads=True, **kw):
            if 'v' in kw.keys():
                del kw['v']
            v = v_fn(self.v)
            if not with_grads:
                v = v.stop_gradients()
            return fn(*a, **kw, v=v)
        new_fn.wrapped = True
        return new_fn

    def _find_variables(self, obj=None):
        vs = Container()
        # ToDo: add support for finding local variables, when JAX supports uniquely flagging variables
        if isinstance(obj, Module) and obj is not self:
            return obj.v
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(v)
                if ret:
                    vs['v' + str(i)] = ret
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ret = self._find_variables(v)
                if ret:
                    vs[k[1:] if k[0] == '_' else k] = ret
            return vs
        elif not hasattr(obj, '__dict__'):
            return vs
        for k, v in obj.__dict__.items():
            if v is not None:
                ret = self._find_variables(v)
                if ret:
                    vs[k[1:] if k[0] == '_' else k] = ret
        return vs

    @staticmethod
    def _extract_v(v, keychain_mappings, orig_key_chain):
        if v.has_key_chain(orig_key_chain):
            ret_cont = v.at_key_chain(orig_key_chain)
        else:
            ret_cont = ivy.Container({})
        for old_kc, new_kc in keychain_mappings.items():
            if orig_key_chain in old_kc:
                ret_cont = ret_cont.set_at_key_chain('/'.join(new_kc.split('/')[1:]), v.at_key_chain(new_kc))
        return ret_cont

    def _wrap_call_methods(self, keychain_mappings, key='', obj=None):
        if isinstance(obj, Module) and obj is not self:
            orig_key_chain = key[1:] if key[0] == '_' else key

            obj.__call__ = self._fn_with_var_arg(obj.__call__,
                                                 lambda v_: self._extract_v(v_, keychain_mappings, orig_key_chain))
            return
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                self._wrap_call_methods(keychain_mappings, key + '/v' + str(i), val)
            return
        elif isinstance(obj, dict):
            for k, val in obj.items():
                k = (key + '/' + k) if key != '' else k
                self._wrap_call_methods(keychain_mappings, k, val)
            return
        if not hasattr(obj, '__dict__'):
            return
        for k, val in obj.__dict__.items():
            k = (key + '/' + k) if key != '' else k
            if val is not None:
                self._wrap_call_methods(keychain_mappings, k, val)
        return

    @staticmethod
    def _remove_duplicate_variables(vs):
        vs_ids = vs.map(lambda x, kc: id(x))
        ids = dict()
        duplicate_keychains = list()
        keychain_mappings = dict()

        def unique_callback(x, kc):
            ids[x] = kc

        def found_dup_callback(x, kc):
            duplicate_keychains.append(kc)
            keychain_mappings[kc] = ids[x]

        vs_ids.map(lambda x, kc: unique_callback(x, kc) if x not in ids else found_dup_callback(x, kc))
        for dup_kc in duplicate_keychains:
            vs = vs.prune_key_chain(dup_kc)
        return vs, keychain_mappings

    def _find_and_create_variables(self, v):
        if v is not None:
            return Container(v)
        vs = Container(dict(**self._find_variables(self), **self._create_variables(self._dev_str)))
        vs, keychain_mappings = self._remove_duplicate_variables(vs)
        self._wrap_call_methods(keychain_mappings, obj=self)
        return vs

    # Overridable #

    def _create_variables(self, dev_str):
        """
        create internal trainable variables, and return as arbitrary nested dict. Overridable.

        :param dev_str: The device string, specifying the device on which to create the variables.
        :type dev_str: string
        """
        return {}

    def _build(self, *args, **kwargs):
        """
        Build the internal layers and variables for this module. Overridable.
        """
        return

    # Abstract #

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        """
        Forward pass of the layer, called after handling the optional input variables.
        """
        raise NotImplementedError

    # Public #
    # -------#

    def __call__(self, *args, v=None, with_grads=True, **kwargs):
        """
        the forward pass of the layer, treating layer instance as callable function.
        """
        if not self._built:
            self.build()
        if v is not None:
            v_orig = self.v
            if not with_grads:
                v = v.stop_gradients()
            self.v = Container(v)
            res = self._forward(*args, **kwargs)
            self.v = v_orig
            return res
        elif hasattr(self.__call__, 'wrapped'):
            return self.__call__(*args, with_grads=with_grads, **kwargs)
        elif not with_grads:
            v_orig = self.v
            self.v = v_orig.stop_gradients()
            ret = self._forward(*args, **kwargs)
            self.v = v_orig
            return ret
        return self._forward(*args, **kwargs)

    def save_weights(self, weights_path):
        """
        Save the weights on the Module.
        :param weights_path: The hdf5 file for saving the weights.
        :type weights_path: string
        """
        os.makedirs('/'.join(weights_path.split('/')[:-1]), exist_ok=True)
        self.v.to_disk_as_hdf5(weights_path)

    def build(self, *args, v=None, **kwargs):
        """
        Build the internal layers and variables for this module.
        """
        self._build(*args, **kwargs)
        self._built = True
        self.v = self._find_and_create_variables(ivy.default(v, self.v))
