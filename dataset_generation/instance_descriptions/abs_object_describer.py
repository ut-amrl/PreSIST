

from abc import ABC, abstractmethod

class AbstractObjectDescriber(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def describeObjects(self, image, instances):
        pass