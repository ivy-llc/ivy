# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithGeneral(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.general as general
        ContainerBase.add_instance_methods(self, general, ['inplace_update', 'unstack', 'gather', 'gather_nd'])
