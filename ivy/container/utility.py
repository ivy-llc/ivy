# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithUtility(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.utility as utility
        ContainerBase.add_instance_methods(self, utility)
