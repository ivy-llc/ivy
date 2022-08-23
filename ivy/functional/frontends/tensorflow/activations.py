import ivy


def exponential(x):
    return ivy.exp(x)


exponential.unsupported_dtypes = {"torch": ("float16",)}
