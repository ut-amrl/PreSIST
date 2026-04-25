

class SemanticClassToPermanenceBucketMapper:
    def __init__(self, class_to_permanence_bucket_mapping):
        self.class_to_permanence_bucket_mapping = class_to_permanence_bucket_mapping

    def map_class_to_permanence_bucket(self, semantic_class):
        return self.class_to_permanence_bucket_mapping.get(semantic_class, None)