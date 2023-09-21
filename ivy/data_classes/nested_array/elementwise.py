# global
from typing import Optional, Union

# local
import ivy
from .base import NestedArrayBase


class NestedArrayElementwise(NestedArrayBase):
    @staticmethod
    def static_add(
        x1: Union[NestedArrayBase, ivy.Array, ivy.NestedArray],
        x2: Union[NestedArrayBase, ivy.Array, ivy.NestedArray],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[ivy.Array] = None,
    ) -> NestedArrayBase:
        pass
        # return self._elementwise_op(other, ivy.add)
