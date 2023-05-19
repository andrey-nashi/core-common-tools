class Label:

    def __init__(self, id: int, name: str, confidence: float = 1.0):
        self.id = id
        self.name = name
        self.confidence = confidence


class LabelSet:

    def __init__(self):
        self.table_labels = []
        self.table_index_forward = {}
        self.table_index_backward = {}

    def add(self, label: Label):
        if label.name is None:
            raise RuntimeException
        self.table_labels[label.name] = label
        index = len(self.table_labels)

        self.table_index_forward[index] = label.name
        self.table_index_backward[label.name] = index

        return index

    def get_label_by_name(self, name: str) -> Label:
        if name in self.table_labels:
            return self.table_labels[name]
        return None

    def get_label_by_index(self, index: int) -> Label:
        if index in self.table_index_forward:
            label_name =  self.table_index_forward[index]
            return self.table_labels[label_name]
        return None

