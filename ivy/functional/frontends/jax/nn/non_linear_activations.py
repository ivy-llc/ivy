import ivy


def relu(x):
    return ivy.relu(x)


relu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
