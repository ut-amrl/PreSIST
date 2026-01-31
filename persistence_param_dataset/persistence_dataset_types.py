from abc import ABC, abstractmethod
from torch import Tensor

class ImageWithMetadata:

    def __init__(self, filename, timeAndDate):
        self.filename = filename
        self.timeAndDate = timeAndDate

class AbsPersistenceParameterOutput(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def toDict(self):
        raise NotImplementedError("Subclasses must implement this method.")


class DatasetResult:

    def __init__(self,
        imageWithMetadata: ImageWithMetadata,
        instanceMasks: dict[int, Tensor], # Instance masks is a dictionary of id -> boolean tensor in the dimensions of the image
        persistenceParameterEstimates: dict[int, AbsPersistenceParameterOutput] # Persistence parameter estimates is a dictionary of id -> persistence parameter output. Ids must match those in instanceMasks
    ):
        self.imageWithMetadata = imageWithMetadata
        self.instanceMasks = instanceMasks
        self.persistenceParameterEstimates = persistenceParameterEstimates


