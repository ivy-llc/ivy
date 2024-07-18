def tensorflow_as_native_dev(device: str, /):
    if isinstance(device, str) and "/" in device:
        return device
    ret = f"/{str(device).upper()}"
    if not ret[-1].isnumeric():
        ret += ":0"
    return ret
