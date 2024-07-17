from .tensorflow__helpers import tensorflow__calculate_fan_in_and_fan_out


def tensorflow__calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    fan_in, fan_out = tensorflow__calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out
