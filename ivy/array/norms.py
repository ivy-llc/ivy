# global
from typing import Optional, List
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithNorms(abc.ABC):
    def layer_norm(
        self: ivy.Array,
        normalized_idxs: List[int],
        epsilon: float = ivy._MIN_BASE,
        scale: float = None,
        offset: float = None,
        new_std: float = 1.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.layer_norm(
            self, normalized_idxs, epsilon, scale, offset, new_std, out=out
        )
