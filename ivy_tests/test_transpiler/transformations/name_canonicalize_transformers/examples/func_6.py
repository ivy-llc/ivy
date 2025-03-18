from torch.distributions.constraints import simplex, real
from torch.nn.functional import pad, conv2d, linear

import torch


def torch_operations():
    y = simplex.check(2) or real.check(1)
    yy = torch.distributions.constraints.simplex.check(
        10
    ) or torch.distributions.constraints.real.check(100)

    z = torch.nn.functional.pad(
        torch.tensor([1, 2, 3]), (1, 1), mode="constant", value=0
    )
    zz = pad(torch.tensor([11, 22, 33]), (1, 1), mode="constant", value=0)

    r = torch.conv2d(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    rr = conv2d(torch.tensor([11, 22, 33]), torch.tensor([11, 22, 33]))
    rrr = torch.nn.functional.conv2d(
        torch.tensor([111, 222, 333]), torch.tensor([111, 222, 333])
    )

    s = torch.nn.functional.linear(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    ss = linear(torch.tensor([11, 22, 33]), torch.tensor([11, 22, 33]))
