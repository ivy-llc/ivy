from torch.distributions.constraints import simplex, real
import torch


def torch_func():
    y = simplex.check(2) or real.check(1)
    yy = torch.distributions.constraints.simplex.check(
        10
    ) or torch.distributions.constraints.real.check(100)

    z = torch.nn.functional.pad(
        torch.tensor([1, 2, 3]), (1, 1), mode="constant", value=0
    )

    r = torch.conv2d(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    rr = torch.nn.functional.conv2d(
        torch.tensor([11, 22, 33]), torch.tensor([11, 22, 33])
    )

    s = torch.nn.functional.linear(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
