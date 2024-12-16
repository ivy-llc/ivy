from math import sin, cos

PI = 3.14159


def custom_sin(x):
    return sin(x) * cos(x) * 2


class MathOperations:
    @staticmethod
    def custom_cos(x):
        return cos(x) * 2


MATH_CONSTANT = 2.71828

MATH_OP = MathOperations()
