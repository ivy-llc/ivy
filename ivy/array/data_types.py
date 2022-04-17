# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithDataTypes(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.data_type as data_type
        ArrayBase.__init__(self, data_type)
