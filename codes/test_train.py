# An example of reading MNIST and training either the MLP baseline or the CNN model.
import argparse
import gzip
import os
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


def read_mnist(data_dir):
    train_images_path = os.path.join(data_dir, "MNIST", "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "MNIST", "train-labels-idx1-ubyte.gz")

    with gzip.open(train_images_path, "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(train_labels_path, "rb") as f:
        _, num = unpack(">2I", f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return train_imgs, train_labs


def build_model(name, input_dim, num_classes, weight_decay):
    if name == "mlp":
        return nn.models.Model_MLP([input_dim, 600, num_classes], "ReLU", [weight_decay, weight_decay])
    if name == "cnn":
        return nn.models.Model_CNN(num_classes=num_classes, lambda_list=[weight_decay, weight_decay])
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--optimizer", choices=["sgd", "momentum"], default="sgd")
    parser.add_argument("--data-dir", default="./dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="./best_models")
    parser.add_argument("--seed", type=int, default=309)
    parser.add_argument("--plot-path", default="./training_curve.png")
    parser.add_argument("--valid-size", type=int, default=10000)
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    train_imgs, train_labs = read_mnist(args.data_dir)

    idx = np.random.permutation(np.arange(train_imgs.shape[0]))
    with open("idx.pickle", "wb") as f:
        pickle.dump(idx, f)

    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:args.valid_size]
    valid_labs = train_labs[:args.valid_size]
    train_imgs = train_imgs[args.valid_size:]
    train_labs = train_labs[args.valid_size:]
    if args.train_limit is not None:
        train_imgs = train_imgs[:args.train_limit]
        train_labs = train_labs[:args.train_limit]

    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0

    model = build_model(args.model, train_imgs.shape[-1], int(train_labs.max() + 1), args.weight_decay)
    if args.optimizer == "momentum":
        optimizer = nn.optimizer.MomentGD(init_lr=args.lr, model=model, mu=0.9)
    else:
        optimizer = nn.optimizer.SGD(init_lr=args.lr, model=model)

    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(train_labs.max() + 1))
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler,
    )

    runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=args.epochs,
        log_iters=100,
        save_dir=args.save_dir,
    )

    _, axes = plt.subplots(1, 2)
    axes.reshape(-1)
    _.set_tight_layout(1)
    plot(runner, axes)
    plt.savefig(args.plot_path)
    print(f"best validation accuracy: {runner.best_score:.5f}")
    print(f"training curve saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
