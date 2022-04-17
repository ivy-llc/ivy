# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithNorms(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.norms as norms
        ArrayBase.__init__(self, norms)
