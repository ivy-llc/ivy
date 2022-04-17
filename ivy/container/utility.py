# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithUtility(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.utility as utility
        ContainerBase.__init__(self, utility)
