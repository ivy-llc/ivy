# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithLayers(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.layers as layers
        ContainerBase.add_instance_methods(self, layers)
