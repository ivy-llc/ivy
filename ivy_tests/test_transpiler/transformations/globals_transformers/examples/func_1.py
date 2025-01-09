from ...mock_dir.custom_math.advanced_math import custom_sin, MATH_OP, MATH_CONSTANT, PI


def math_func(x):
    return custom_sin(x) + MATH_OP.square(x) + PI * MATH_CONSTANT
