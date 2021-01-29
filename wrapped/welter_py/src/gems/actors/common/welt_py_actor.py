from abc import ABC, abstractmethod


class Actor(ABC):
    def __init__(self, index,mode = None):
        self.index = index
        self.mode = mode
        super().__init__()

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def invoke(self):
        pass

    @abstractmethod
    def terminate(self):
        pass