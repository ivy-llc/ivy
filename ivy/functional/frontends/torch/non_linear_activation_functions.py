# local
import ivy


def relu(input):
    return ivy.relu(input)


relu.unsupported_dtypes = ("float16",)
