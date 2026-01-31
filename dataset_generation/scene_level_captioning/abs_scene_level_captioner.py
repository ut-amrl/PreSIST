

from abc import ABC, abstractmethod

class AbstractSceneLevelCaptioner(ABC):

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    @abstractmethod
    def generateCaption(self, image):
        pass