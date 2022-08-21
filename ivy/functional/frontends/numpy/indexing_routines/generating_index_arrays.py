import ivy


def nonzero(x, /):
    if x==None:
        # numpy nonzero returns an empty array when a is None, same as when a is empty
        x = []
    return ivy.nonzero(x)
