# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSorting(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.sorting as sorting
        ContainerBase.add_instance_methods(self, sorting)
