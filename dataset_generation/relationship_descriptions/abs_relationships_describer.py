from abc import ABC, abstractmethod

class AbsRelationshipDescriber(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def describeRelationships(self, image, instances, objectDescriptions):
        pass