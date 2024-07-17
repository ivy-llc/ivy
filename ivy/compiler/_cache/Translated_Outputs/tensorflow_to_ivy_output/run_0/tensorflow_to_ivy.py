import tensorflow
import tensorflow as tf

from typing import Union
from typing import Iterable
from typing import Optional
from typing import Dict

from .tensorflow__helpers import tensorflow__to_ivy
from .tensorflow__helpers import tensorflow_nested_map


def tensorflow_to_ivy(
    x: Union[tensorflow.Tensor, tf.Tensor, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[str, bool]] = None,
):
    if nested:
        return tensorflow_nested_map(
            tensorflow__to_ivy, x, include_derived, shallow=False
        )
    return tensorflow__to_ivy(x)
