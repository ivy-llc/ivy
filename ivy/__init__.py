from .core import *
from . import neural_net
from .neural_net import *
from . import verbosity
from .framework_handler import get_framework, set_framework, unset_framework, framework_stack


class Array:

    def __init__(self):
        pass

    @property
    def shape(self):
        return

    @property
    def dtype(self):
        return


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
