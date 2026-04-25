import re


class ClosedSetSemanticClassList:
    """
    Loads a closed set of semantic classes from a file or list.

    Supports two line formats:
      - "N: name"  (e.g. COCO-Stuff: "62: chair") — IDs are taken from the file.
      - "name"     (plain text) — IDs are auto-assigned as 0-indexed positions.
    """

    _ID_NAME_RE = re.compile(r"^(\d+):\s*(.+)$")

    def __init__(self, class_text_file: str | None, classes_list: list[str] | None = None):
        if (class_text_file is None) == (classes_list is None):
            raise ValueError("Exactly one of class_text_file or classes_list must be provided")

        if class_text_file is not None:
            self.id_to_class, self.class_to_id, self.semantic_classes = self._load_classes(class_text_file)
        else:
            self.semantic_classes = list(classes_list)
            self.id_to_class = {i: name for i, name in enumerate(self.semantic_classes)}
            self.class_to_id = {name: i for i, name in enumerate(self.semantic_classes)}

    @classmethod
    def _load_classes(cls, class_text_file: str) -> tuple[dict[int, str], dict[str, int], list[str]]:
        id_to_class: dict[int, str] = {}
        class_to_id: dict[str, int] = {}
        names: list[str] = []

        with open(class_text_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            m = cls._ID_NAME_RE.match(line)
            if m:
                class_id, name = int(m.group(1)), m.group(2).strip()
            else:
                class_id, name = i, line

            id_to_class[class_id] = name
            class_to_id[name] = class_id
            names.append(name)

        return id_to_class, class_to_id, names
