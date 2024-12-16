from ...mock_dir.data_utils.preprocessing import normalize, Preprocessor
from ...mock_dir.data_utils.analysis import analyze_data, DATA_THRESHOLD


def process_data(data):
    normalized = normalize(data)
    scaled = Preprocessor.scale(normalized)
    return analyze_data(scaled) > DATA_THRESHOLD
