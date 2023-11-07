# local
import ivy
import ivy.functional.frontends.torch as torch_frontend


_default_dtype = torch_frontend.float32


def can_cast(from_, to):
    from_str = str(from_)
    to_str = str(to)
    if "float" in from_str and "bool" in to_str:
        return False
    if "float" in from_str and "int" in to_str:
        return False
    if "uint" in from_str and ("int" in to_str and "u" not in to_str):
        if ivy.dtype_bits(to) < ivy.dtype_bits(from_):
            return False
    if "complex" in from_str and ("float" in to_str or "int" in to_str):
        return False
    if "bool" in to_str:
        return from_str == to_str
    return True


def get_default_dtype():
    return _default_dtype


def promote_types(type1, type2, /):
    return torch_frontend.promote_types_torch(type1, type2)


def set_default_dtype(d):
    ivy.utils.assertions.check_elem_in_list(
        d,
        [
            torch_frontend.float64,
            torch_frontend.float32,
            torch_frontend.float16,
            torch_frontend.bfloat16,
        ],
        message="only floating-point types are supported as the default type",
    )
    global _default_dtype
    _default_dtype = d
    return
