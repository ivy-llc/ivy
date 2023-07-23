"""Module providingFunction of activation of selu."""
import ivy
def selu(inp, lambdaa = 1.0507, alpha = 1.6732):
    """Function of SeLU activation"""
    y =1
    if type(inp)==type(y):
        if inp >= 0:
            return ivy.array(int(lambdaa * inp))
        else:

            return ivy.array(int(lambdaa * alpha * (ivy.exp(inp) - 1)))
    shape = ivy.Shape((len(inp),))
    fill_value = ivy.Container(a=1)
    dtype = ivy.Container(a=ivy.int64)
    arr = ivy.full(shape, fill_value,dtype=dtype)
    arr= arr['a']
    for i in range(len(inp)):
        if inp[i] >= 0:
            arr[i] = int(lambdaa * inp[i])
        else:
             arr[i] = int(lambdaa * alpha * (ivy.exp(inp[i]) - 1))
    return arr
