import tensorflow
import tensorflow as tf

from typing import Tuple
from typing import Dict
from typing import List
from typing import Union
from typing import Iterable

from .tensorflow__helpers import tensorflow_get_item


def tensorflow_index_nest(
    nest: Union[List, Tuple, Dict, tensorflow.Tensor, tf.Tensor, dict],
    index: Union[List[int], Tuple[int], Iterable[int]],
    /,
):
    ret = nest
    for i in index:
        ret = tensorflow_get_item(ret, i)
    return ret
