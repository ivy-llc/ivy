# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


class broadcast:
    @to_ivy_arrays_and_back
    def __init__(self, *args):
        data = ivy.broadcast_arrays(*map(ivy.array, args))
        self.__shape = data[0].shape
        self.__ndim = data[0].ndim
        self.__index = 0
        self.__numiter = len(data)
        self.__size = data[0].size
        self.__data = tuple((*zip(*(ivy.flatten(i) for i in data)),))
        self.__iters = tuple((iter(ivy.flatten(i)) for i in data))

    @property
    def shape(self):
        return self.__shape

    @property
    def ndim(self):
        return self.__ndim

    @property
    def nd(self):
        return self.__ndim

    @property
    def numiter(self):
        return self.__numiter

    @property
    def size(self):
        return self.__size

    @property
    def iters(self):
        return self.__iters

    @property
    def index(self):
        return self.__index

    def __next__(self):
        if self.index < self.size:
            self.__index += 1
            return self.__data[self.index - 1]
        raise StopIteration

    def __iter__(self):
        return self

    def reset(self):
        self.__index = 0
