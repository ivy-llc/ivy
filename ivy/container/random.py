# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithRandom(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.random as random
        ContainerBase.add_instance_methods(self, random, to_ignore=['shuffle'])
