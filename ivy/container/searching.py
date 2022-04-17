# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithSearching(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.searching as searching
        ContainerBase.__init__(self, searching)
