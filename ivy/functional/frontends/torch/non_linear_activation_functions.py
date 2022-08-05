# local
import ivy


def relu(input):
    return ivy.relu(input)


relu.unsupported_dtypes = ("float16",)



def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)

