import tensorflow
import numpy as np

from typing import TypeVar
from typing import Optional
from typing import Union

from .tensorflow_NestedSequence_bknd import tensorflow_NestedSequence_bknd
from .tensorflow__helpers import tensorflow__asarray_infer_dtype_bknd
from .tensorflow__helpers import tensorflow__asarray_to_native_arrays_and_back_bknd
from .tensorflow__helpers import tensorflow_as_native_dev
from .tensorflow__helpers import tensorflow_dev
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion

SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")


@tensorflow_handle_array_like_without_promotion
@tensorflow__asarray_to_native_arrays_and_back_bknd
@tensorflow__asarray_infer_dtype_bknd
def tensorflow_asarray(
    obj: Union[
        tensorflow.Tensor,
        tensorflow.Variable,
        tensorflow.TensorShape,
        bool,
        int,
        float,
        tensorflow_NestedSequence_bknd,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[tensorflow.DType] = None,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    with tensorflow.device(device):
        if tensorflow.is_tensor(obj):
            ret = tensorflow.cast(obj, dtype) if obj.dtype != dtype else obj
        elif (
            dtype is not None
            and dtype.is_integer
            and np.issubdtype(np.array(obj).dtype, np.floating)
        ):
            obj_np = np.array(obj)
            ret = tensorflow.convert_to_tensor(obj_np, dtype)
        else:
            ret = tensorflow.convert_to_tensor(obj, dtype)
        return (
            tensorflow.identity(ret)
            if copy or tensorflow_as_native_dev(tensorflow_dev(ret)) != device
            else ret
        )
