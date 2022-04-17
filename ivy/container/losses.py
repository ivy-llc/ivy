# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithLosses(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.losses as losses
        ContainerBase.__init__(self, losses)
