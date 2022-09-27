# local
import ivy


# def is_storage(obj):
# 	return ivy.is_storage(obj)

# def is_complex(obj):
# 	return ivy.is_complex(obj)

# def is_conj(obj):
# 	return ivy.is_conj(obj)

# def is_nonzero(obj):
# 	return ivy.is_nonzero(obj)

# def set_flush_denormal(obj):
# 	ivy.set_flush_denormal(obj)

# def set_default_dtype(obj):
# 	ivy.set_default_dtype(obj)

# def set_default_tensor_type(obj):
# 	ivy.set_default_tensor_type(obj)


def numel(input):
    num = 1
    input_shape = ivy.asarray(input).shape
    for e in input_shape:
        num = num * e
    return num


def is_floating_point(input):
    return ivy.is_float_dtype(ivy.asarray(input).dtype)
