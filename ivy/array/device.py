# local
from ivy.array.base import ArrayBase

# ToDo: implement all methods here as public instance methods


class ArrayWithDevice(ArrayBase):

    def __init__(self):
        import ivy.functional.ivy.device as device
        ArrayBase.__init__(self, device)
