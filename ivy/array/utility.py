# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithUtility(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.utility as utility
        ArrayBase.__init__(self, utility)
