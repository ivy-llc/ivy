# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithRandom(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.random as random
        ArrayBase.__init__(self, random)
