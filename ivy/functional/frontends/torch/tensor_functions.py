# local
import ivy


def is_tensor(obj):
    return ivy.is_array(obj)


# def is_storage(obj):
# 	return ivy.is_storage(obj)

# def is_complex(obj):
# 	return ivy.is_complex(obj)

# def is_conj(obj):
# 	return ivy.is_conj(obj)


def numel(input):
    ivy.assertions.check_true(
        is_tensor(input),
        message="input must be a tensor",
    )
    return input.size


def is_floating_point(input):
    ivy.assertions.check_true(
        is_tensor(input),
        message="input must be a tensor",
    )
    return ivy.is_float_dtype(input)


def is_nonzero(input):
    ivy.assertions.check_true(
        is_tensor(input),
        message="input must be a tensor",
    )
    ivy.assertions.check_equal(
        numel(input),
        1,
        message="bool value of tensor with more than one or no values is ambiguous",
    )
    if input.ndim:
        return not input[0]
    else:
        return not input
