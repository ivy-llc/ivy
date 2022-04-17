# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithActivations(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.activations as activations
        ContainerBase.add_instance_methods(self, activations)
