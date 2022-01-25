"""
PyTorch Pointwise Ops Frontend
"""

# local
import ivy


# noinspection PyShadowingBuiltins
def abs(input, *, out=None):
    ret = ivy.abs(input)
    if ivy.exists(out):
        ivy.assert_supports_inplace(out)
        return ivy.inplace_update(out, ret)
    return ret
