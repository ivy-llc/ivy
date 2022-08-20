# global
import ivy


def map(f, xs):
    return ivy.stack([f(x) for x in xs])
