# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithCreation(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.creation as creation
        ArrayBase.__init__(self, creation)
