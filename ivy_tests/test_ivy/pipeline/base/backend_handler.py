from abc import ABC, abstractmethod


class BackendHandler(ABC):
    @abstractmethod
    def set_backend(self, backend: str):
        pass

    @abstractmethod
    def unset_backend(self):
        pass
