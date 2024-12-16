from ...mock_dir.data_utils.preprocessing import Preprocessor
from ...mock_dir.ml_models.neural_net import NeuralNetwork


def nested_operation(data):
    preprocessor = Preprocessor()
    model = NeuralNetwork([5, 3, 1])
    return model.forward(preprocessor.scale(data))
