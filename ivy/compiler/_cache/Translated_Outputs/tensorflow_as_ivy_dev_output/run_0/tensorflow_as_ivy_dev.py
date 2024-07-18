from .tensorflow__helpers import tensorflow_split_1


def tensorflow_as_ivy_dev(device: str, /):
    if isinstance(device, str) and "/" not in device:
        return str(device)
    dev_in_split = tensorflow_split_1(device[1:], ":")[-2:]
    if len(dev_in_split) == 1:
        return str(dev_in_split[0])
    dev_type, dev_idx = dev_in_split[0], dev_in_split[1]
    dev_type = dev_type.lower()
    if dev_type == "cpu":
        return str(dev_type)
    return str(f"{dev_type}:{dev_idx}")
