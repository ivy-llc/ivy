# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithLosses(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.losses as losses
        ArrayBase.__init__(self, losses)
