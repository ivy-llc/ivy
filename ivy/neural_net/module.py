"""
Base class for deriving trainable modules
"""

# global
import abc

# local
from ivy.core.container import Container


# Base #
# -----#

class Module(abc.ABC):

    def __init__(self, dev_str, v=None):
        """
        Initialze Ivy layer, which is a stateful object consisting of trainable variables.

        :param dev_str: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
        :type dev_str: str
        :param v: Ivy container of trainable variables. Created internally by default.
        :type v: ivy container, optional
        """
        self._dev_str = dev_str
        if v is None:
            self.v = Container(self._find_and_create_variables())
        else:
            self.v = Container(v)

    # Private #
    # --------#

    def _fn_with_var_arg(self, fn, key):
        def new_fn(*a, **kw):
            return fn(*a, **kw, v=self.v[key])
        new_fn.wrapped = True
        return new_fn

    def _find_variables(self, key='', obj=None):
        obj = self if obj is None else obj
        vs = dict()
        # ToDo: add support for finding local variables, when JAX supports uniquely flagging variables
        if isinstance(obj, Module) and obj is not self:
            vs[key[1:] if key[0] == '_' else key] = obj.v
            obj.__call__ = self._fn_with_var_arg(obj.__call__, key[1:] if key[0] == '_' else key)
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
            vs = dict(**vs, **self._find_variables(k, val))
        return vs

    def _find_and_create_variables(self):
        return dict(**self._find_variables(), **self._create_variables(self._dev_str))

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
