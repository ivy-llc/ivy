from numpy import array_function_dispatch
from numpy.core.fromnumeric import _take_dispatcher, _wrapfunc


@array_function_dispatch(_take_dispatcher)
def take(a, indices, axis=None, mode='raise'):
    return _wrapfunc(a, 'take', indices, axis=axis, mode=mode)
