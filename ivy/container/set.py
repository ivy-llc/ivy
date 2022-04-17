# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSet(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.set as set
        self.add_instance_methods(set)
