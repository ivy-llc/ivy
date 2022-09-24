import ivy


def bilinear(input1, input2, weight, bias=None):
    input1_shape = ivy.shape(input1)
    if len(input1_shape) < 2:
        raise RuntimeError("Input1 dimensions must be of format (N,*,Hin1)")
    input2_shape = ivy.shape(input2)
    if len(input2_shape) < 2:
        raise RuntimeError("Input2 dimensions must be of format (N,*,Hin2)")
    if input1_shape[:-1] != input2.shape[:-1]:
        raise RuntimeError("All dimension of input1 and input2 \
            should match except for last dimension")
    weight_shape = ivy.shape(weight)
    if len(weight_shape) < 3:
        raise RuntimeError("Weight dimensions must be of format (Hout,Hin1,Hin2)")
    if weight_shape[-2] != input1_shape[-1]:
        raise RuntimeError("Weight 2nd last dimension \
            should match input1 last dimension")
    if weight_shape[-1] != input2_shape[-1]:
        raise RuntimeError("Weight last dimension should match input2 last dimension")
    if ivy.shape(bias) != weight_shape[:-2]:
        raise RuntimeError("Bias dimension should match weight first dimension")

    # Reshape to (*, Hin1)
    input1_flattened = ivy.reshape(input1, (-1, input1_shape[-1]))
    # Reshape to (*, Hin2)
    input2_flattened = ivy.reshape(input2, (-1, input2_shape[-1]))
    # Reshape to (Hout, Hin1, Hin2)
    weight_flattened = ivy.reshape(weight, (-1, weight_shape[-2], weight_shape[-1]))

    # Create empty output array with shape (*, Hout)
    output = ivy.zeros((ivy.shape(weight_flattened)[0]  ,ivy.shape(input1_flattened)[0]))

    for i in range(ivy.shape(weight_flattened)[0]):
        buff_1 = ivy.matmul(input1_flattened, weight_flattened[i])
        ivy.multiply(buff_1, input2_flattened, out=buff_1)
        buff_2 = ivy.sum(buff_1, axis=-1)
        # Should be replaced by ivy.narrow
        for j in range(ivy.shape(output)[0]):
            output[j][i] = buff_2[j]

    if bias is not None:
        # Reshape to (Hout)
        bias_flattened = ivy.reshape(bias, -1)
        output = ivy.add(output, bias_flattened)

    # Reshape output to original shape (N,*,Hout)
    ivy.reshape(output, input1_shape[:-1] + weight_shape[:-2])
    return output
