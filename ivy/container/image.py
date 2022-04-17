# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithImage(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.image as image
        ContainerBase.__init__(self, image)
