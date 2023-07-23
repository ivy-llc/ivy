import ivy
import tensorflow
import jax
import numpy
def SELU(x, lambdaa = 1.0507, alpha = 1.6732):
    y =1
    if type(x)==type(y):
        if x >= 0:
            return ivy.array(int(lambdaa * x))
        else:

            return ivy.array(int(lambdaa * alpha * (ivy.exp(x) - 1)))
    elif len(x)==0:
        if x >= 0:
            return ivy.array(int(lambdaa * x))
        else:

            return ivy.array(int(lambdaa * alpha * (ivy.exp(x) - 1)))
    shape = ivy.Shape((len(x),))
    fill_value = ivy.Container(a=1)
    dtype = ivy.Container(a=ivy.int32)
    arr = ivy.full(shape, fill_value,dtype=dtype)
    arr= arr['a']
    for i in range(len(x)):
        if x[i] >= 0:
            arr[i] = int(lambdaa * x[i])
        else:

             arr[i] = int(lambdaa * alpha * (ivy.exp(x[i]) - 1))
    return arr
