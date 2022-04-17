# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithManipulation(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.manipulation as manipulation
        ArrayBase.__init__(self, manipulation)
