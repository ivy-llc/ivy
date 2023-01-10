import torch


def l2_normalize(x: torch.Tensor,
                 axis: int = None,
                 out: torch.Tensor = None
                 ) -> torch.Tensor:
    if axis is None:
        axis = tuple(range(x.ndim))
    return torch.nn.functional.normalize(x, p=2, dim=axis, out=out)


l2_normalize.support_native_out = True
