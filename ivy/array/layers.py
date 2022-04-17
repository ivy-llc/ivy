# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithLayers(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.layers as layers
        ArrayBase.__init__(self, layers)
