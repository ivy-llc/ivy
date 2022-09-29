# global

# local
import ivy


def result_type(*arg1):
    res=ivy.dtype(ivy.abs(sum(arg1)))
    return res

