from .tensorflow__helpers import tensorflow_get_item


def tensorflow__handle_padding_shape(padding, n, mode):
    ag__result_list_0 = []
    for i in range(int(len(padding) / 2) - 1, -1, -1):
        res = (
            tensorflow_get_item(padding, i * 2),
            tensorflow_get_item(padding, i * 2 + 1),
        )
        ag__result_list_0.append(res)
    padding = tuple(ag__result_list_0)
    if mode == "circular":
        padding = padding + ((0, 0),) * (n - len(padding))
    else:
        padding = ((0, 0),) * (n - len(padding)) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding
