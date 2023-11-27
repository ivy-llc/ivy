from abc import ABC, abstractmethod


class Criterion(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def reverse_reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, new_pos):
        raise NotImplementedError
