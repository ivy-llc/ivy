import tensorflow

from typing import Union
from typing import Optional

from .tensorflow__helpers import tensorflow_copy_array
from .tensorflow__helpers import tensorflow_size_1
from .tensorflow__helpers import tensorflow_stack_1
from .tensorflow__helpers import tensorflow_to_ivy
from .tensorflow__helpers import tensorflow_unstack_1


def tensorflow_copy_array(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    *,
    to_ivy_array: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if isinstance(x, tensorflow.TensorArray):
        x_wrapped = tensorflow_stack_1(x)
        y = tensorflow.TensorArray(x.dtype, tensorflow_size_1(x)())
        x = tensorflow_unstack_1(y, tensorflow_copy_array(x_wrapped))
    else:
        x = tensorflow.identity(x)
    if to_ivy_array:
        return tensorflow_to_ivy(x)
    return x
