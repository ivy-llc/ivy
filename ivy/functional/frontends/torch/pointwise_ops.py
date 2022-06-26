# global
import ivy

def tan(input, *, out=None):
    return ivy.tan(input, out=out)

tan.unsupported_dtypes = ('float16',)
