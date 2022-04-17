# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithSet(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.set as set
        ArrayBase.__init__(self, set)
