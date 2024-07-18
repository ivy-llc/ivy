import tensorflow as tf

from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_get_item


def tensorflow_promote_types(
    type1: Union[str, tf.DType],
    type2: Union[str, tf.DType],
    /,
    *,
    array_api_promotion: bool = False,
):
    if not (type1 and type2):
        return type1 if type1 else type2
    query = [tensorflow_as_ivy_dtype_1(type1), tensorflow_as_ivy_dtype_1(type2)]
    query = tuple(query)
    if query not in promotion_table:
        query = query[1], query[0]

    def _promote(query):
        if array_api_promotion:
            return tensorflow_get_item(array_api_promotion_table, query)
        return tensorflow_get_item(promotion_table, query)

    return _promote(query)
