"""
PyTorch Pointwise Ops Frontend
"""

# local
import ivy


# noinspection PyShadowingBuiltins
def abs(input, *, out=None):
    # ToDo: support inplace operations once ivy.inplace_update is implemented
    return ivy.abs(input)
