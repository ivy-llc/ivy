import ivy_tests.test_transpiler.transformations.mock_dir.custom_math.advanced_math as adv_math
from ...mock_dir.data_utils import preprocessing as pp
from ...mock_dir.ml_models.neural_net import NeuralNetwork


def complex_operation(data):
    normalized = pp.normalize(data)
    sin_value = adv_math.custom_sin(normalized)
    model = NeuralNetwork([5, 3, 1])
    return model.forward(sin_value) + adv_math.MATH_CONSTANT
