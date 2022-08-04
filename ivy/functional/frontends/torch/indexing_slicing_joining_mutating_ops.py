# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)

def concat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)
