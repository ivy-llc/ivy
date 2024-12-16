from ...mock_dir.ml_models.neural_net import NeuralNetwork, MODEL_VERSION


def create_model():
    model = NeuralNetwork([10, 5, 1])
    print(f"Model version: {MODEL_VERSION}")
    return model
