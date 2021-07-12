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

    def __init__(self, dev_str=None, v=None):
        """
        Initialze Ivy layer, which is a stateful object consisting of trainable variables.

        :param dev_str: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
        :type dev_str: str, optional
        :param v: Ivy container of trainable variables. Created internally by default.
        :type v: ivy container, optional
        """
        if dev_str is None:
            dev_str = 'gpu:0' if ivy.gpu_is_available() else 'cpu'
        self._dev_str = dev_str
        if v is None:
            self.v = self._find_and_create_variables()
        else:
            self.v = Container(v)

    # Private #
    # --------#

    def _fn_with_var_arg(self, fn, v_fn):
        def new_fn(*a, **kw):
            if 'v' in kw.keys():
                del kw['v']
            return fn(*a, **kw, v=v_fn(self.v))
        new_fn.wrapped = True
        return new_fn

    def _find_variables(self, key='', obj=None):
        obj = self if obj is None else obj
        vs = dict()
        # ToDo: add support for finding local variables, when JAX supports uniquely flagging variables
        if isinstance(obj, Module) and obj is not self:
            vs[key[1:] if key[0] == '_' else key] = obj.v
            return vs
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                vs = dict(**vs, **self._find_variables(key + str(i), v))
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                k = (key + '/' + k) if key != '' else k
                vs = dict(**vs, **self._find_variables(k, v))
            return vs
        if not hasattr(obj, '__dict__'):
            return vs
        for k, val in obj.__dict__.items():
            k = (key + '/' + k) if key != '' else k
            if val is not None:
                vs = dict(**vs, **self._find_variables(k, val))
        return vs

    @staticmethod
    def _extract_v(v, keychain_mappings, orig_key):
        if orig_key in v.keys():
            ret_cont = v[orig_key]
        else:
            ret_cont = ivy.Container({})
        for old_kc, new_kc in keychain_mappings.items():
            if orig_key in old_kc:
                ret_cont = ret_cont.set_at_key_chain(new_kc, v.at_key_chain(new_kc))
        return ret_cont

    def _wrap_call_methods(self, keychain_mappings, key='', obj=None):
        # ToDo: check whether keychain_mappings need to be recursively refined for checks in the sub-calls
        obj = self if obj is None else obj
        if isinstance(obj, Module) and obj is not self:
            orig_key = key[1:] if key[0] == '_' else key

            obj.__call__ = self._fn_with_var_arg(obj.__call__,
                                                 lambda v_: self._extract_v(v_, keychain_mappings, orig_key))
            return
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._wrap_call_methods(keychain_mappings, key + str(i), v)
            return
        elif isinstance(obj, dict):
            for k, v in obj.items():
                k = (key + '/' + k) if key != '' else k
                self._wrap_call_methods(keychain_mappings, k)
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

    def _find_and_create_variables(self):
        vs = Container(dict(**self._find_variables(), **self._create_variables(self._dev_str)))
        vs, keychain_mappings = self._remove_duplicate_variables(vs)
        self._wrap_call_methods(keychain_mappings)
        return vs

    # Overridable #

    def _create_variables(self, dev_str):
        """
        create internal trainable variables, and return as arbitrary nested dict.

        :param dev_str: The device string, specifying the device on which to create the variables.
        :type dev_str: string
        """
        return {}

    # Abstract #

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        """
        the forward pass of the layer, called after handling the optional input variables.
        """
        raise NotImplementedError

    # Public #
    # -------#

    def __call__(self, *args, v=None, **kwargs):
        """
        the forward pass of the layer, treating layer instance as callable function.
        """
        if v is not None:
            v_orig = self.v
            self.v = Container(v)
            res = self._forward(*args, **kwargs)
            self.v = v_orig
            return res
        if hasattr(self.__call__, 'wrapped'):
            return self.__call__(*args, **kwargs)
        return self._forward(*args, **kwargs)

    def save_weights(self, weights_path):
        """
        Save the weights on the Module.
        :param weights_path: The hdf5 file for saving the weights.
        :type weights_path: string
        """
        os.makedirs('/'.join(weights_path.split('/')[:-1]), exist_ok=True)
        self.v.to_disk_as_hdf5(weights_path)
