# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithSearching(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.searching as searching
        ArrayBase.__init__(self, searching)
