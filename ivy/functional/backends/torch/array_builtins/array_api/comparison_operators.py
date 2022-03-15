import torch

def builtin_lt(self: torch.Tensor,other: torch.Tensor)\
        -> torch.Tensor :
    if hasattr(self,'dtype') and hasattr(other,'dtype'):
        promoted_type = torch.promote_types(self.dtype,other.dtype)
        self = self.to(promoted_type)
        other = other.to(promoted_type)
    return self.__lt__(other)