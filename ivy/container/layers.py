# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithLayers(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.layers as layers
        ContainerBase.__init__(self, layers)
