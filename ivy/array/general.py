# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithGeneral(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.general as general
        ArrayBase.__init__(self, general)
