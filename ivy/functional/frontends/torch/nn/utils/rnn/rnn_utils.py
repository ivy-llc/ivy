import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


def pack_padded_sequence(
    input,
    lengths,
    batch_first=False,
    enforce_sorted=True,
):
    raise ivy.exceptions.IvyNotImplementedException(
        "torch.nn.utils.rnn.pack_padded_sequence is not implemented in torch frontend"
    )


def pack_sequence(
    sequences,
    enforce_sorted=True,
):
    raise ivy.exceptions.IvyNotImplementedException(
        "torch.nn.utils.rnn.pack_sequence is not implemented in torch frontend"
    )


def pad_packed_sequence(
    sequence,
    batch_first=False,
    padding_value=0.0,
    total_length=None,
    lengths=None,
):
    raise ivy.exceptions.IvyNotImplementedException(
        "torch.nn.utils.rnn.pad_packed_sequence is not implemented in torch frontend"
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


def unpack_sequence(
    sequence,
):
    raise ivy.exceptions.IvyNotImplementedException(
        "torch.nn.utils.rnn.unpack_sequence is not implemented in torch frontend"
    )


def unpad_sequence(sequences, batch_first=False):
    raise ivy.exceptions.IvyNotImplementedException(
        "torch.nn.utils.rnn.unpad_sequence is not implemented in torch frontend"
    )
