# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithDataTypes(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.data_type as data_type
        self.add_instance_methods(data_type)
