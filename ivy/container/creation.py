# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithCreation(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.creation as creation
        self.add_instance_methods(creation)
