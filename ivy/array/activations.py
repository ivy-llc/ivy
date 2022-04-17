# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithActivations(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.activations as activations
        ArrayBase.__init__(self, activations)
