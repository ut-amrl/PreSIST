
import json


from pathlib import Path


PERSISTENCE_FILE_NAME = "persistence_parameters.json"

def readPersistenceFileJson(path: str) -> dict:

    with open(path, 'r', encoding='utf-8') as f:
        persistenceParametersJson = json.load(f)

        return persistenceParametersJson


def readPersistenceData(path: str) -> list:

    persistenceParametersJson = readPersistenceFileJson(path)

    # If the persistenceParameters is a dict and has a key "imagePath" and "persistenceParams", extract the value of "persistenceParameters"
    if isinstance(persistenceParametersJson, dict) and "persistenceParams" in persistenceParametersJson:
        return persistenceParametersJson["persistenceParams"]

    return persistenceParametersJson


def writePersistenceData(persistenceParamsPath: str, persistenceParameters: list, imgFileName: str = None) -> None:
    persistenceParamsPath = Path(persistenceParamsPath)
    persistenceParamsPath.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = persistenceParamsPath.with_suffix(".json.tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        persistenceDataOut = {
            "persistenceParams": persistenceParameters,
            "imagePath": imgFileName
        }
        json.dump(persistenceDataOut, f, indent=2, ensure_ascii=False)
    tmp_path.replace(persistenceParamsPath)