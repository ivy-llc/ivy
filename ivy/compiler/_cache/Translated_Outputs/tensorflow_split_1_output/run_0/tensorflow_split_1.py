from .tensorflow__helpers import tensorflow_get_item
from .tensorflow__helpers import tensorflow_handle_methods_1
from .tensorflow__helpers import tensorflow_split


@tensorflow_handle_methods_1
def tensorflow_split_1(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        split_size_or_sections = [split_size] * (
            tensorflow_get_item(tensor.shape, dim) // split_size
        )
        if tensorflow_get_item(tensor.shape, dim) % split_size:
            split_size_or_sections.append(
                tensorflow_get_item(tensor.shape, dim) % split_size
            )
    return tuple(
        tensorflow_split(
            tensor,
            num_or_size_splits=split_size_or_sections,
            axis=dim,
            with_remainder=True,
        )
    )
