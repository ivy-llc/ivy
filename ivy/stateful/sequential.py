"""Base class for deriving trainable modules."""

# global
from typing import Union, Optional

# local
import ivy
from ivy.stateful.module import Module


class Sequential(Module):
    def __init__(
        self,
        *sub_modules: Module,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        v: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    ):
        """Initialize a sequential container. Modules will be added to it in
        the order they are passed in the constructor.

        Parameters
        ----------
        submodules
            Submodules to chain together into a sequence.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc.
        v
            the variables for each submodule in the sequence, constructed internally by
            default.
        """
        if v is not None:
            for i, submod in enumerate(sub_modules):
                try:
                    submod.v = v["submodules"][f"v{str(i)}"]
                except KeyError as e:
                    if submod.v:
                        raise ivy.utils.exceptions.IvyException(
                            "variables v passed to Sequential class must have key "
                            "chains in the form of "
                            '"submodules/v{}", where {} is an idx'
                        ) from e
        self._submodules = list(sub_modules)
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def __iter__(self):
        return iter(self._submodules)

    def _forward(self, inputs):
        """Perform forward pass of the Sequential container.

        Parameters
        ----------
        inputs
            Inputs to process.

        Returns
        -------
        ret
            The output after each of the layers in the Sequential has been applied.
        """
        x = inputs
        for i, submod in enumerate(self._submodules):
            try:
                x = submod(x, v=self.v.submodules[f"v{str(i)}"])
            except KeyError as e:
                if submod.v:
                    raise ivy.utils.exceptions.IvyException(
                        "variables v passed to Sequential class must have key chains "
                        "in the form of "
                        '"submodules/v{}", where {} is an idx'
                    ) from e
                x = submod(x)
        return x

    def _extra_repr(self):
        submods = []
        for i, submod in enumerate(self._submodules):
            submods.append(f"v{i}={submod}")
        return ", ".join(submods)
