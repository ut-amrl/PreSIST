

class PersistenceParamDataset:

    def __init__(self, datasetConfig):
        self.datasetConfig = datasetConfig


    def getItem(self, index):
        raise NotImplementedError("Still need to implement getItem method.")

    def __len__(self):
        raise NotImplementedError("Still need to implement __len__ method.")