import numpy as np


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


class Preprocessor:
    @staticmethod
    def scale(data):
        return data / np.max(data)


SCALING_FACTOR = 100

PREPROCESS_CONSTANT = 1
