class Cyclic_dependency_class:
    def echo(
        self,
    ):
        a = global_b["B"].echo()
        b = global_a["A"].echo()
        return a, b


def helper_fn():
    a = global_a["A"].echo()
    b = global_b["B"].echo()
    return a, b


def cyclic_dependency_func():
    return helper_fn()


class A:
    def __init__(self):
        pass

    def echo(self):
        return "I am class A"


class B:
    def __init__(self):
        pass

    def echo(self):
        return "I am class B"


GLOB_A = A()
GLOB_B = B()

global_a = {"A": GLOB_A}
global_b = {"B": GLOB_B}
