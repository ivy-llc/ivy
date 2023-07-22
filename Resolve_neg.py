import torch

def resolve_neg(tensor):
    """
    Resolves the negation bit of a tensor.

    Args:
        tensor (torch.Tensor): The tensor to resolve.

    Returns:
        torch.Tensor: The tensor with the negation bit cleared.
    """

    if not isinstance(tensor, torch.Tensor):
        raise ivy.exceptions.IvyError("tensor must be a torch.Tensor")

    output = tensor.clone()
    output.neg = False
    return output

if __name__ == "__main__":
    x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
    y = x.conj()
    z = y.imag

    print(z.is_neg())  # True

    out = resolve_neg(z)

    print(out.is_neg())  # False

    print(out)  # 1.0+1.0j
