# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithImage(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.image as image
        ArrayBase.__init__(self, image)
