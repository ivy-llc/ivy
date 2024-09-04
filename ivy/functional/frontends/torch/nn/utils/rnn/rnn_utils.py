import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def pad_sequence(
    sequences,
    batch_first=False,
    padding_value=0.0,
):
    return ivy.pad_sequence(
        sequences, batch_first=batch_first, padding_value=padding_value
    )
