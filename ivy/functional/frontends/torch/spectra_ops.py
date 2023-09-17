#Add Spectral Ops to PyTorch Frontend #15

import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.torch import promote_types_of_torch_inputs



@to_ivy_arrays_and_back
def stft(signals, frame_length, frame_step,
         fft_length=None, window_fn=None, pad_end=False, name=None,
         ):
    signals = ivy.asarray(signals)
    return ivy.stft(signals, frame_length, frame_step, fft_length=fft_length,
                    window_fn=window_fn, pad_end=pad_end, name=name,
                   )
