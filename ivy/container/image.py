# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithImage(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.image as image
        ContainerBase.add_instance_methods(self, image)
