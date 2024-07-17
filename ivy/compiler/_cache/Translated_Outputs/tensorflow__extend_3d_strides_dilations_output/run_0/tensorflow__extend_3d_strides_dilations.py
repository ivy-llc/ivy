def tensorflow__extend_3d_strides_dilations(strides, dilations, data_format):
    if data_format[-1] == "C":
        strides = [1, *([strides] * 3 if isinstance(strides, int) else strides), 1]
        dilations = [
            1,
            *([dilations] * 3 if isinstance(dilations, int) else dilations),
            1,
        ]
    else:
        strides = [1, 1, *([strides] * 3 if isinstance(strides, int) else strides)]
        dilations = [
            1,
            1,
            *([dilations] * 3 if isinstance(dilations, int) else dilations),
        ]
    return strides, dilations
