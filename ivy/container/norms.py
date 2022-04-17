# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithNorms(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.norms as norms
        ContainerBase.__init__(self, norms)
