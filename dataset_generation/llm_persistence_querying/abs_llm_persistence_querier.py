

from abc import ABC, abstractmethod

class AbsLLMPersistenceQuerier(ABC):

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    @abstractmethod
    def estimatePersistenceParameters(self, sceneCaption, objectDescriptions, relationshipDescriptions, timeAndDate):
        pass