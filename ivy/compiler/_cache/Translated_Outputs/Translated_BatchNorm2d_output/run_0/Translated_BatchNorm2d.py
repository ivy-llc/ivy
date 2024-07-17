from .Translated__BatchNorm import Translated__BatchNorm


class Translated_BatchNorm2d(Translated__BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
