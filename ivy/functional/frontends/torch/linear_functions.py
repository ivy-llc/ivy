import ivy
from typing import Union

def bilinear(
    input1: Union[ivy.Array, ivy.NativeArray],
    input2: Union[ivy.Array, ivy.NativeArray],
    weight: Union[ivy.Array, ivy.NativeArray],
    bias: Union[ivy.Array, ivy.NativeArray] = None,
)-> ivy.Array:
    return ivy.linear(ivy.linear(ivy.matrix_transpose(input1), weight), input2, bias=bias)