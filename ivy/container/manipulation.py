# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithManipulation(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.manipulation as manipulation
        ContainerBase.__init__(self, manipulation)
