# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithGeneral(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.general as general
        ContainerBase.__init__(self, general, ['inplace_update'])
