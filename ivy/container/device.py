# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithDevice(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.device as device
        ContainerBase.__init__(self, device)
