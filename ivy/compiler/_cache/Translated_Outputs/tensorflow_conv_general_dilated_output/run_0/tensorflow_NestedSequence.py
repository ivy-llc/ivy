from typing import Protocol
from typing import TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class tensorflow_NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /): ...

    def __len__(self, /): ...
