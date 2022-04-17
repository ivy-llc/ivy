# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithGradients(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.gradients as gradients
        ContainerBase.__init__(self, gradients)
