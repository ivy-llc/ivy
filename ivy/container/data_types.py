# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithDataTypes(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.data_type as data_type
        ContainerBase.__init__(self, data_type)
