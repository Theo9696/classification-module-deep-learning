class Model:
    def __init__(self, nb_classes_out: int):
        self.model = None
        self.nb_classes_out = nb_classes_out

    def get_parameters(self):
        return {}
