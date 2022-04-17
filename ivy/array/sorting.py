# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithSorting(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.sorting as sorting
        ArrayBase.__init__(self, sorting)
