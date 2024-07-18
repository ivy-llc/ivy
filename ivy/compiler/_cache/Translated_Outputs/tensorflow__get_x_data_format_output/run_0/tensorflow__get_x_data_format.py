def tensorflow__get_x_data_format(dims: int = 2, data_format: str = "channel_first"):
    if dims == 1:
        if data_format == "channel_first":
            return "NCW"
        else:
            return "NWC"
    if dims == 2:
        if data_format == "channel_first":
            return "NCHW"
        else:
            return "NHWC"
    elif dims == 3:
        if data_format == "channel_first":
            return "NCDHW"
        else:
            return "NDHWC"
