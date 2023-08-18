#Add Spectral Ops to PyTorch Frontend #15

import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch import promote_types_of_torch_inputs


@to_ivy_arrays_and_back
def stft(input):
    if len(ivy.shape(input)) > 2:
        raise ivy.exceptions.IvyError(
            "stft(): expected input to have two or fewer dimensions but got an"
            f" input with {ivy.shape(input)} dimansions"
        )
    return ivy.stft(input, y=None, rowvar=True)
        
