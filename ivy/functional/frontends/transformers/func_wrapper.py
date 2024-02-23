# Still working on it

import functools
import torch
from typing import Callable

from transformers import PreTrainedModel, PreTrainedTokenizer


# --- Helpers --- #
# --------------- #


def _from_transformers_tensors(x):
    if isinstance(x, dict):
        return x
    return x


def _to_transformers_tensors(x, tokenizer):
    if isinstance(x, str):
        return tokenizer(x, return_tensors="pt")
    elif isinstance(x, torch.Tensor):
        return x
    return x


# --- Main --- #
# ------------ #


def inputs_to_transformers_tensors(
    fn: Callable, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_transformers_tensors(*args, **kwargs):
        """
        Convert input data into Transformers tensors.

        Convert input data in both the positional and keyword arguments
        into Transformers tensors (PyTorch tensors or dictionaries), and
        then call the function with the updated arguments.
        """
        # Convert input data to Transformers tensors
        new_args = [_to_transformers_tensors(arg, tokenizer) for arg in args]
        new_kwargs = {
            _to_transformers_tensors(key, tokenizer): value
            for key, value in kwargs.items()
        }

        return fn(*new_args, **new_kwargs)

    return _inputs_to_transformers_tensors


def outputs_to_pytorch_tensors(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_pytorch_tensors(*args, **kwargs):
        """
        Convert Transformers tensors to PyTorch tensors.

        Call the function, and then convert all Transformers tensors
        (PyTorch tensors or dictionaries) returned by the function to
        PyTorch tensors.
        """
        # Call the unmodified function
        ret = fn(*args, **kwargs)

        # Convert output data to PyTorch tensors
        return _from_transformers_tensors(ret)

    return _outputs_to_pytorch_tensors


def to_transformers_tensors_and_back(
    fn: Callable, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Callable:
    """
    Wrap `fn` to work with Transformers tensors.

    Wrap `fn` so that input data is converted to Transformers tensors
    (PyTorch tensors or dictionaries), and output data is converted to
    PyTorch tensors.
    """
    return outputs_to_pytorch_tensors(
        inputs_to_transformers_tensors(fn, model, tokenizer)
    )
