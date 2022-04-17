# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithManipulation(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.manipulation as manipulation
        self.add_instance_methods(manipulation, to_ignore=['expand_dims', 'split', 'repeat', 'swapaxes', 'reshape'])
