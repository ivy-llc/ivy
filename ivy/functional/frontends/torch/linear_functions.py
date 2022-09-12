import ivy


def bilinear(input1, input2, weight, bias=None):
    return ivy.linear(
        ivy.linear(ivy.matrix_transpose(input1), weight), 
        input2, 
        bias=bias
    )
    


