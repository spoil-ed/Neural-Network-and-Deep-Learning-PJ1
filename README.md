# Neural Network and Deep Learning PJ1

Course project for MNIST classification using a small neural-network framework implemented with NumPy.

Repository: https://github.com/spoil-ed/Neural-Network-and-Deep-Learning-PJ1

## Contents

- `codes/mynn/`: NumPy implementations of layers, models, optimizers, scheduler, runner, and metrics.
- `codes/test_train.py`: train MLP/CNN models.
- `codes/test_model.py`: evaluate a saved checkpoint.
- `codes/test_core_ops.py`: unit tests for core operators.
- `codes/report/main.pdf`: final project report.
- `codes/report/main.tex`: LaTeX source of the report.

## Implemented Parts

- Part A: MLP baseline with linear forward/backward and softmax cross-entropy.
- Part B: CNN model with self-implemented `conv2D`.
- Part C: Momentum optimization and L2 weight decay regularization.

## Run

```bash
cd codes
python3 -m unittest test_core_ops.py
python3 test_train.py --model mlp --epochs 5
python3 test_train.py --model cnn --optimizer momentum --epochs 5 --lr 0.03
python3 test_model.py --model-path ./best_models/best_model.pickle --model-type auto
```

The MNIST dataset and trained checkpoints are not included in this repository.

## Model Checkpoints

Trained checkpoints are hosted on ModelScope:

https://modelscope.cn/models/imspoiled/Neural-Network-and-Deep-Learning-PJ1-Checkpoint
