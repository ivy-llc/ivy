# global
import ivy

# slogdet


def slogdet(a):
    sign, logabsdet = ivy.slogdet(a)
    return ivy.concat((ivy.reshape(sign, (-1,)), ivy.reshape(logabsdet, (-1,))))


slogdet.unsupported_dtypes = ("float16",)
