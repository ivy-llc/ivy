# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithStatistical(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.statistical as statistical
        ContainerBase.add_instance_methods(self, statistical, to_ignore=['einsum'])
