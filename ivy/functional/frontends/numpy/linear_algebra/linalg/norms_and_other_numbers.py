# global
import ivy

# slogdet


def slogdet(a):
    sign, logabsdet = ivy.slogdet(a)
    ret = ivy.concat((ivy.reshape(sign, (-1,)), ivy.reshape(logabsdet, (-1,))))
    return ret


slogdet.unsupported_dtypes = ("float16",)
