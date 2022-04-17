# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithStatistical(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.statistical as statistical
        self.add_instance_methods(statistical, to_ignore=['einsum'])
