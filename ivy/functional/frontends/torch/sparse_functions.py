import ivy


def one_hot(tensor, num_classes=-1):
    if num_classes == -1:
        depth = int(tensor.max() + 1)
    else:
        depth = num_classes
    return ivy.one_hot(tensor, depth)


one_hot.supported_dtypes = {"torch": ("int64",)}
