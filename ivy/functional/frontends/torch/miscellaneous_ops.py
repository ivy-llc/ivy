import ivy


def flip(input, dims):
    return ivy.flip(input, axis=dims)


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def fliplr(input):
    assert len(input.shape) >= 2, "Requires the tensor to be at least 2-D"
    return ivy.flip(input, axis=(-1,))


def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, axis=dims)


roll.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def cumsum(input, dim, *, dtype=None, out=None):
    return ivy.cumsum(x=input, axis=dim, out=out)


cumsum.unsupported_dtypes = (
    "float16",
    "uint16",
    "uint32",
    "uint64",
)


def cumprod(input, dim, *, dtype=None, out=None):
    return ivy.cumprod(x=input, axis=dim, out=out)


cumprod.unsupported_dtypes = (
    "float16",
    "uint16",
    "uint32",
    "uint64",
)


def diagonal(input, offset=0, dim1=0, dim2=1):
    return ivy.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


def block_diag(*tensors):
    total_size_x = sum(
        [tensor.shape[-1] if len(tensor.shape) > 0 else 1 for tensor in tensors]
    )
    total_size_y = sum(
        [tensor.shape[-2] if len(tensor.shape) > 1 else 1 for tensor in tensors]
    )

    data = ivy.zeros((total_size_y, total_size_x), dtype=ivy.result_type(*tensors))

    current_x_start = 0
    current_y_start = 0

    for tensor in tensors:
        x_size = tensor.shape[-1] if len(tensor.shape) > 0 else 1
        y_size = tensor.shape[-2] if len(tensor.shape) > 1 else 1

        for x in range(x_size):
            for y in range(y_size):
                # It feels like there should be a much faster way of doing this
                # but that might be my C programming instinct kicking in
                # because the first thing that comes to mind to fix that is
                # to do a short string of memcpy operations and some
                # sketchy pointer arithmetic.

                # Equally the performance may well be fine anyway I haven't checked.
                if len(tensor.shape) == 2:
                    data[current_y_start + y][current_x_start + x] = tensor[y][x]
                elif len(tensor.shape) == 1:
                    data[current_y_start + y][current_x_start + x] = tensor[x]
                elif len(tensor.shape) == 0:
                    data[current_y_start + y][current_x_start + x] = tensor

        current_y_start += y_size
        current_x_start += x_size

    return data
