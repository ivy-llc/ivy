# local
from typing import Tuple
import ivy


def shape(array, /) -> Tuple:
    return ivy.shape(array)