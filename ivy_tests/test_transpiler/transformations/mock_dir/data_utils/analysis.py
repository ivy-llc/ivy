from .preprocessing import normalize, SCALING_FACTOR


def analyze_data(data):
    return normalize(data).sum()


DATA_THRESHOLD = 0.5
