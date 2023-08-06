import json
import struct
import sys

import numpy as np
import torch


def export(pth_file_path: str, output_dir_path: str):
    model = torch.load(pth_file_path, map_location="cpu")
    with open(f"{output_dir_path}/rwkv.bin", "wb") as bin_file, \
            open(f"{output_dir_path}/rwkv.json", "w") as json_file:
        model_info = {}
        for name, weights in model.items():
            flattened_weights = weights.float().numpy().astype(np.float32).flatten()
            # Weights
            bin_file.write(struct.pack(f"{len(flattened_weights)}f", *flattened_weights))
            # Information about layer names and shapes for reference
            model_info[name] = list(weights.shape)
        json.dump(model_info, json_file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 exporter.py <PTH_FILE_PATH> <OUTPUT_DIR_PATH>")
        exit(1)
    _, pth_file_path, output_dir_path = sys.argv[:3]
    export(pth_file_path, output_dir_path)
