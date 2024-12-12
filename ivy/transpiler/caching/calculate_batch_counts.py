import json
import sys


def calculate_batch_count(matrix, batch_size):
    total_items = len(matrix)
    batch_count = (total_items + batch_size - 1) // batch_size
    return batch_count


def main(matrix_json, batch_size):
    matrix = json.loads(matrix_json)
    batch_count = calculate_batch_count(matrix, batch_size)
    print(batch_count)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
