# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithStatistical(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.statistical as statistical
        ContainerBase.__init__(self, statistical, to_ignore=['einsum'])
