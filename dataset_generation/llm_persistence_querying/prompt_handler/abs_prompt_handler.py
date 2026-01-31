from abc import ABC, abstractmethod

class AbsPromptHandler(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generatePrompt(self, sceneCaption, objectDescriptions, relationshipDescriptions, timeAndDate):
        pass

    @abstractmethod
    def parsePersistenceParamsFromLLMResponse(self, responseText):
        pass