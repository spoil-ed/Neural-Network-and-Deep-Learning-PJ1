import unittest

import numpy as np

import mynn as nn


class CoreOperatorTests(unittest.TestCase):
    def test_linear_forward_backward(self):
        layer = nn.op.Linear(2, 3, initialize_method=lambda size: np.ones(size))
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = layer(x)
        np.testing.assert_allclose(out, np.array([[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]))

        upstream = np.ones_like(out) * 0.5
        dx = layer.backward(upstream)
        np.testing.assert_allclose(layer.grads["W"], np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
        np.testing.assert_allclose(layer.grads["b"], np.array([[1.0, 1.0, 1.0]]))
        np.testing.assert_allclose(dx, np.ones((2, 2)) * 1.5)

    def test_cross_entropy_loss_and_grad(self):
        class DummyModel:
            def __init__(self):
                self.received = None

            def backward(self, grads):
                self.received = grads
                return grads

        model = DummyModel()
        loss = nn.op.MultiCrossEntropyLoss(model=model, max_classes=2)
        logits = np.array([[2.0, 1.0], [0.5, 1.5]])
        labels = np.array([0, 1])

        value = loss(logits, labels)
        self.assertAlmostEqual(value, 0.3132616875, places=7)
        grad = loss.backward()
        self.assertEqual(grad.shape, logits.shape)
        np.testing.assert_allclose(model.received, grad)
        self.assertAlmostEqual(float(grad.sum(axis=1).max()), 0.0, places=7)

    def test_conv2d_forward_backward_shape(self):
        layer = nn.op.conv2D(1, 1, 2, initialize_method=lambda size: np.ones(size))
        x = np.arange(9.0).reshape(1, 1, 3, 3)
        out = layer(x)
        np.testing.assert_allclose(out, np.array([[[[9.0, 13.0], [21.0, 25.0]]]]))

        dx = layer.backward(np.ones_like(out))
        self.assertEqual(dx.shape, x.shape)
        np.testing.assert_allclose(layer.grads["W"], np.array([[[[8.0, 12.0], [20.0, 24.0]]]]))
        np.testing.assert_allclose(layer.grads["b"], np.array([4.0]))

    def test_sgd_updates_parameters_in_place(self):
        model = nn.models.Model_MLP([2, 1], "ReLU")
        layer = model.layers[0]
        layer.W[...] = 1.0
        layer.b[...] = 0.0
        layer.grads["W"] = np.ones_like(layer.W)
        layer.grads["b"] = np.ones_like(layer.b)
        old_w_ref = layer.W

        nn.optimizer.SGD(0.1, model).step()
        self.assertIs(layer.W, old_w_ref)
        np.testing.assert_allclose(layer.W, np.ones_like(layer.W) * 0.9)


if __name__ == "__main__":
    unittest.main()
