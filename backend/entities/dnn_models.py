class DnnModel:
    """Represents a Deep Neural Network model with its specific attributes."""

    idx = 0

    def __init__(
        self,
        name,
        level,
        sum_activation,
        model_size,
        input_shape,
        accuracy,
        model_flops,
        type,
        nb_layers,
        params=0.0,
        weighted_sum_neurons=0.0,
    ):
        DnnModel.idx += 1
        self.id = DnnModel.idx
        self.vehicule = None
        self.server = None
        self.level = level
        self.sum_activation = sum_activation
        self.params = params
        self.model_size = model_size  # Bytes
        self.input_shape = input_shape
        self.accuracy = accuracy
        self.model_flops = model_flops
        self.name = name
        self.type = type
        self.nb_layers = nb_layers
        self.weighted_sum_neurons = weighted_sum_neurons

    def save_as_dict(self) -> dict:
        return {
            "name": self.name,
            "vehicule": self.vehicule,
            "server": self.server,
            "model_size": self.model_size,
            "input_shape": self.input_shape,
            "accuracy": self.accuracy,
            "model_flops": self.model_flops,
            "type": self.type,
        }
