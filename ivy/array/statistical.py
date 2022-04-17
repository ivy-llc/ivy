# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithStatistical(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.statistical as statistical
        ArrayBase.__init__(self, statistical)
