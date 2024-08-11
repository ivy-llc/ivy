from typing import TypeVar
from typing import Protocol

_T_co = TypeVar("_T_co", covariant=True)


class tensorflow_NestedSequence_bknd(Protocol[_T_co]):
    def __getitem__(self, key: int, /): ...

    def __len__(self, /): ...
