# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithNorms(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.norms as norms
        ContainerBase.add_instance_methods(self, norms)
