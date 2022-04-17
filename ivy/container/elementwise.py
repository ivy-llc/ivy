# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyUnresolvedReferences,PyMissingConstructor
class ContainerWithElementwise(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.elementwise as elementwise
        self.add_instance_methods(elementwise)
