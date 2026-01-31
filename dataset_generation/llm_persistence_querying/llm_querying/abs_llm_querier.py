from abc import ABC, abstractmethod

class AbsLLMQuerier(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def queryLLM(self, prompt):
        pass


