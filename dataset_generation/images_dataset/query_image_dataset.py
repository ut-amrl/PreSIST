from persistence_param_dataset.persistence_dataset_types import ImageWithMetadata

class AbsQueryImageDataset:

    def __init__(self):
        pass

    def getImagesWithMetadata(self) -> list[ImageWithMetadata]:
        raise NotImplementedError("Subclasses must implement this method.")
