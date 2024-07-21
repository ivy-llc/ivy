def Translated__no_grad_fill_(tensor, val):
    return tensor.fill_(val)


def Translated_ones_(tensor):
    return Translated__no_grad_fill_(tensor, 1.0)


def Translated__no_grad_zero_(tensor):
    return tensor.zero_()


def Translated_zeros_(tensor):
    return Translated__no_grad_zero_(tensor)
