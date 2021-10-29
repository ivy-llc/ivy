wrapping_paused = False
op_logging = False
wrapped_stack = list()


ARRAY_BUILTINS = ['__neg__', '__pow__', '__rpow__', '__add__', '__radd__', '__iadd__', '__sub__', '__rsub__',
                  '__isub__', '__mul__', '__rmul__', '__imul__', '__truediv__', '__rtruediv__', '__itruediv__',
                  '__floordiv__', '__rfloordiv__', '__ifloordiv__', '__abs__', '__lt__', '__le__', '__eq__', '__ne__',
                  '__gt__', '__ge__', '__and__', '__rand__', '__or__', '__ror__', '__invert__', '__xor__', '__rxor__',
                  '__getitem__', '__setitem__', '__getattr__', '__setattr__', '__getattribute__']

CLASSES_TO_WRAP = {'numpy': [],
                   'jax': [],
                   'tensorflow': [],
                   'torch': [('torch', 'Tensor')],
                   'mxnet': []}

GRAPH_ATTRIBUTES = {'numpy': [],
                   'jax': [],
                   'tensorflow': [],
                   'torch': ['data', 'requires_grad'],
                   'mxnet': []}
