from dataset_generation.relationship_descriptions.abs_relationships_describer import AbsRelationshipDescriber
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload

class EmptyRelationshipDescriber(AbsRelationshipDescriber):

    def __init__(self, name="empty_relationship_describer"):
        super().__init__(name)

    def describeRelationships(self, image, instances, objectDescriptions):
        # Return an empty list of relationships
        return {}, DebugPayload.empty()