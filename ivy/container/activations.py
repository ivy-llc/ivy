# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithActivations(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.activations as activations
        ContainerBase.__init__(self, activations)
