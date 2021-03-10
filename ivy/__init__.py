from .core import *
from . import neural_net as nn
from .neural_net import *
from . import verbosity
from .framework_handler import get_framework, set_framework, unset_framework, framework_stack


class Tensor:

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


backend = 'none'
