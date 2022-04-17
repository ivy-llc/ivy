# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithLinearAlgebra(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.linear_algebra as linear_algebra
        ArrayBase.__init__(self, linear_algebra)
