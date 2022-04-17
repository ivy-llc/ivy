# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithCreation(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.creation as creation
        ContainerBase.__init__(self, creation)
