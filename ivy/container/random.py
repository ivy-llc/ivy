# local
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class ContainerWithRandom(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.random as random
        ContainerBase.__init__(self, random, to_ignore=['shuffle'])
