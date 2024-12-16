from ...mock_dir.data_utils.preprocessing import PREPROCESS_CONSTANT


def scale_data(data):
    return data * PREPROCESS_CONSTANT * GLOB_2 + GLOB_1


GLOB_1 = 10

GLOB_2 = GLOB_1
