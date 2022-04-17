# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithSet(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.set as set
        ContainerBase.__init__(self, set)
