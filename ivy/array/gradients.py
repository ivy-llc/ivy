# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithGradients(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.gradients as gradients
        ArrayBase.__init__(self, gradients)
