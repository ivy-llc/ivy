# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.elementwise as elementwise
        ArrayBase.__init__(self, elementwise)
