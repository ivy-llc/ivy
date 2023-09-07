# global
from typing import Literal

# local
import ivy
from ..array.array import Array


class QArray(Array):
    def __init__(self,
                 array: ivy.Array,
                 dtype: ivy.Dtype,
                 /,
                 qscheme: Literal['naive', 'affine']=None
        ):
        super().__init__(array.data)
        self._original_dtype = ivy.dtype(array)
        self._qscheme = qscheme
        self._scale_factor = None
        self._zero_point = None
        self.__quantize__(qscheme, dtype)

    @property
    def qscheme(self):
        return self._qscheme
    
    @property
    def scale(self):
        return self._scale_factor
    
    @property
    def zero_point(self):
        return self._zero_point
    
    def __quantize_naive__(self, dtype: ivy.Dtype):
        self.data = ivy.astype(self.data, dtype)

    def __quantize_affine__(self, dtype: ivy.Dtype):
        """
        Calculates the scale factor and the zero_point values of a model then applies affine quantization to weights
        """
        def round_half_up(x):
            mask = (x >= 0)
            out = ivy.empty_like(x)
            out[mask] = ivy.floor(x[mask] + 0.5)
            out[~mask] = ivy.ceil(x[~mask] - 0.5)
            return out
        
        def round_up(x):
            return ivy.floor(x + 0.5)

        bits = ivy.dtype_bits(dtype)
        beta = ivy.min(self).item()
        alpha = ivy.max(self).item()
        delta = alpha - beta
        self._scale_factor = (2**bits - 1) / delta
        self._zero_point = -1 * round(beta*self._scale_factor) - 2**(bits-1)
        scaled_data = ivy.clip(self.data, beta, alpha)
        scaled_data = ivy.nested_map(scaled_data, lambda x: round_up(self._scale_factor * x + self._zero_point))
        scaled_data = ivy.astype(scaled_data, dtype)
        self.data = scaled_data.data

    def __quantize__(self, qscheme: Literal['naive', 'affine'], dtype: ivy.Dtype):
        # Default values for qscheme
        if qscheme is None:
            if isinstance(dtype, type(ivy.float16)):
                qscheme = "naive"
            elif isinstance(dtype, type(ivy.int8)):
                qscheme = "affine"

        if qscheme == "naive":
            self.__quantize_naive__(dtype)

        elif qscheme == "affine":
            self.__quantize_affine__(dtype)
            

    def dequantize(self) -> ivy.Array:
        ret_data = ivy.copy(self.data)
        if self.qscheme == "naive":
            return ivy.astype(ivy.asarray(self.data), ivy.float32)
        elif self.qscheme == "affine":
            # TODO: Implement affine dequantize
            return ret_data
        return ret_data

    def fallback(self):
        pass
        
# def quantize_array(obj, dtype) -> QArray:
#     if isinstance(dtype, ivy.float16):
#         pass