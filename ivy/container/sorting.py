# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithSorting(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.sorting as sorting
        ContainerBase.__init__(self, sorting)
