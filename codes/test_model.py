import argparse
import gzip
import os
import pickle
from struct import unpack

import numpy as np

import mynn as nn


def read_test_mnist(data_dir):
    test_images_path = os.path.join(data_dir, "MNIST", "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "MNIST", "t10k-labels-idx1-ubyte.gz")

    with gzip.open(test_images_path, "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(test_labels_path, "rb") as f:
        _, num = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return test_imgs / 255.0, test_labs


def load_model(path, model_type):
    if model_type == "auto":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model_type = "cnn" if isinstance(payload, dict) and payload.get("type") == "Model_CNN" else "mlp"

    if model_type == "cnn":
        model = nn.models.Model_CNN()
    else:
        model = nn.models.Model_MLP()
    model.load_model(path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./best_models/best_model.pickle")
    parser.add_argument("--model-type", choices=["auto", "mlp", "cnn"], default="auto")
    parser.add_argument("--data-dir", default="./dataset")
    args = parser.parse_args()

    model = load_model(args.model_path, args.model_type)
    test_imgs, test_labs = read_test_mnist(args.data_dir)
    logits = model(test_imgs)
    print(nn.metric.accuracy(logits, test_labs))


if __name__ == "__main__":
    main()
