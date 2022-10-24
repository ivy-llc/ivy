import ivy


@to_ivy_arrays_and_back
def split(a, splits, axis=0):
    return ivy.split(a, splits=None, axis=0)
