# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.device as device
        self.add_instance_methods(device)
