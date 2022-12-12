# local
import ivy


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
    if "bool" in to_str:
        return from_str == to_str
    return True
