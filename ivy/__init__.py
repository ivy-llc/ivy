from .core import *
from . import neural_net_functional
from .neural_net_functional import *
from . import neural_net_stateful
from .neural_net_stateful import *
from . import verbosity
from .framework_handler import current_framework, get_framework, set_framework, unset_framework, framework_stack


class Array:

    def __init__(self):
        pass

    @property
    def shape(self):
        return

    @property
    def dtype(self):
        return

    def __getitem__(self):
        pass


class Variable:

    def __init__(self):
        pass

    @property
    def shape(self):
        return

    @property
    def dtype(self):
        return


class Framework:

    def __init__(self):
        pass


class Device:

    def __init__(self):
        pass


class Dtype:

    def __init__(self):
        pass


backend = 'none'
