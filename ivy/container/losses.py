# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithLosses(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.losses as losses
        self.add_instance_methods(losses)
