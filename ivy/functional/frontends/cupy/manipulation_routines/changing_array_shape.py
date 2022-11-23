# local
import ivy


def reshape(x, /, newshape, order="C"):
    return ivy.reshape(x, shape=newshape)
