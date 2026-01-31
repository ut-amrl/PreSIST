from abc import ABC, abstractmethod

class AbsIntermediateResultsLogger(ABC):

    @abstractmethod
    def logInputData(self, image, timeAndDate, inputBatchInfo):
        pass

    @abstractmethod
    def logSegmentationResults(self, segmentationResults):
        pass

    @abstractmethod
    def logSceneCaption(self, caption):
        pass

    @abstractmethod
    def logObjectDescriptions(self, objectDescriptions):
        pass

    @abstractmethod
    def logRelationshipDescriptions(self, relationshipDescriptions):
        pass

    @abstractmethod
    def logPrompts(self, prompts):
        pass

    @abstractmethod
    def logPersistenceParameters(self, persistenceParameters):
        pass