def tensorflow__extend_2d_padding(padding, data_format):
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        padding = [(padding, padding)] * 2
    if data_format[-1] == "C":
        padding = [(0, 0)] + padding + [(0, 0)]
    else:
        padding = [(0, 0), (0, 0)] + padding
    return padding
