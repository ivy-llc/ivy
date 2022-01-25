"""
PyTorch Pointwise Ops Frontend
"""

# local
import ivy


# noinspection PyShadowingBuiltins
def abs(input, *, out=None):
    ret = ivy.abs(input)
    if ivy.exists(out):
        if ivy.is_variable(out):
            if not ivy.inplace_variables_supported():
                raise Exception(
                    'Inplace variables are not supported for {} backend'.format(ivy.current_framework_str()))
        elif ivy.is_array(out):
            if not ivy.inplace_arrays_supported():
                raise Exception(
                    'Inplace arrays are not supported for {} backend'.format(ivy.current_framework_str()))
        return ivy.inplace_update(out, ret)
    return ret
