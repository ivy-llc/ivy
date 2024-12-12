import os
import dill
import logging


def merge_cache_files():
    base_dir = "/ivy"
    all_caches_dir = os.path.join(base_dir, "tracer-transpiler", "all_cache_artifacts")
    cache_dir = os.path.join(
        base_dir, "tracer-transpiler", "ivy_repo", "ivy", "compiler", "_cache"
    )

    os.makedirs(cache_dir, exist_ok=True)

    logging.debug(f"Base directory: {base_dir}")
    logging.debug(f"All caches directory: {all_caches_dir}")
    logging.debug(f"Cache directory: {cache_dir}")

    pkl_files = [
        "torch_to_torch_frontend_translation_cache.pkl",
        "torch_frontend_to_ivy_translation_cache.pkl",
        "ivy_to_tensorflow_translation_cache.pkl",
    ]

    logging.debug(f"Pickle files to process: {pkl_files}")

    for pkl_file in pkl_files:
        merged_dict = {}
        logging.debug(f"Processing {pkl_file}")
        for root, _, files in os.walk(all_caches_dir):
            if pkl_file in files:
                file_path = os.path.join(root, pkl_file)
                logging.debug(f"Found {pkl_file} in {file_path}")
                with open(file_path, "rb") as f:
                    try:
                        data = dill.load(f)
                    except Exception as e:
                        logging.debug(e)
                        logging.debug(file_path)
                        continue
                    merged_dict.update(data)

        output_path = os.path.join(cache_dir, pkl_file)
        with open(output_path, "wb") as f:
            dill.dump(merged_dict, f)
        logging.debug(f"Merged cache saved to {output_path}")


if __name__ == "__main__":
    merge_cache_files()
