# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithLinearAlgebra(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.linear_algebra as linear_algebra
        ContainerBase.add_instance_methods(self, linear_algebra, to_ignore=['matrix_norm'])
