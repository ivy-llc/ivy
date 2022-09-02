# global
import ivy

# slogdet


def slogdet(a):
    return ivy.slogdet(a)

slogdet.unsupported_dtypes = ("float16",)