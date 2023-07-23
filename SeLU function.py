import ivy

def SELU(x, lambdaa = 1.0507, alpha = 1.6732):
    if x >= 0:
        return ivy.array(int(lambdaa * x))
    else:
        return ivy.array(int(lambdaa * alpha * (ivy.exp(x) - 1)))