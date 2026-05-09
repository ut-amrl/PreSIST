import json

from persistence_eval.core.domain import PersistenceParameterEstimate


class SemanticClassPersistenceParamLookup:
    """
    Maps integer class IDs to PersistenceParameterEstimate objects.
    JSON files use string representations of the int IDs as keys (standard JSON practice).
    """

    def __init__(self, params_file: str | None = None):
        self.id_to_params: dict[int, PersistenceParameterEstimate] = {}
        if params_file is not None:
            self.load_params_from_file(params_file)

    def get_params_for_id(self, class_id: int) -> PersistenceParameterEstimate:
        if class_id not in self.id_to_params:
            raise KeyError(f"Class ID {class_id} not found in parameter lookup.")
        return self.id_to_params[class_id]

    def set_params_for_id(self, class_id: int, params: PersistenceParameterEstimate) -> None:
        self.id_to_params[class_id] = params

    def load_params_from_file(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)
        result: dict[int, PersistenceParameterEstimate] = {}
        for class_id, entry in data.items():
            try:
                key = int(class_id)
            except ValueError:
                # Non-integer keys (e.g. the "__global__" fallback entry the
                # location-keyed trainer writes alongside class ids) are not
                # meaningful in a class-keyed lookup; skip them silently so the
                # same params JSON works for both class and location consumers.
                continue
            result[key] = PersistenceParameterEstimate(
                params=entry["params"],
                confidence=entry.get("confidence"),
                metadata=entry.get("metadata", {}),
            )
        self.id_to_params = result

    def save_params_to_file(self, file_path: str) -> None:
        data = {
            str(class_id): {
                "params": estimate.params,
                "confidence": estimate.confidence,
                "metadata": estimate.metadata,
            }
            for class_id, estimate in self.id_to_params.items()
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
