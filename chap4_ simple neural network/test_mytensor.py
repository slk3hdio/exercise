import importlib.util
import unittest
from pathlib import Path

import numpy as np

import torch


from MyTensor import MyTensor


class TestMyTensorAgainstPyTorch(unittest.TestCase):
    def assertArrayClose(self, actual, expected, atol=1e-6, rtol=1e-6):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def test_elementwise_ops_with_broadcast_backward(self):
        x_data = np.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=float)
        y_data = np.array([[2.0, -1.0, 0.5]], dtype=float)

        x = MyTensor(x_data)
        y = MyTensor(y_data)
        out = ((x + y) * (x - 2.0)).mean()
        out.backward()

        tx = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
        ty = torch.tensor(y_data, dtype=torch.float64, requires_grad=True)
        tout = ((tx + ty) * (tx - 2.0)).mean()
        tout.backward()

        self.assertArrayClose(out.data, tout.detach().numpy())
        self.assertArrayClose(x.grad, tx.grad.numpy())
        self.assertArrayClose(y.grad, ty.grad.numpy())

    def test_matmul_relu_sum_backward(self):
        a_data = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=float)
        b_data = np.array([[2.0, -1.0], [0.0, 3.0], [1.0, -4.0]], dtype=float)

        a = MyTensor(a_data)
        b = MyTensor(b_data)
        out = (a @ b).relu().sum()
        out.backward()

        ta = torch.tensor(a_data, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b_data, dtype=torch.float64, requires_grad=True)
        tout = (ta @ tb).relu().sum()
        tout.backward()

        self.assertArrayClose(out.data, tout.detach().numpy())
        self.assertArrayClose(a.grad, ta.grad.numpy())
        self.assertArrayClose(b.grad, tb.grad.numpy())

    def test_transpose_and_power_backward(self):
        x_data = np.array([[1.5, -2.0], [0.5, 3.0], [-1.0, 2.0]], dtype=float)

        x = MyTensor(x_data)
        out = ((x.T * x.T).mean() + x.pow(3).mean())
        out.backward()

        tx = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
        tout = ((tx.T * tx.T).mean() + tx.pow(3).mean())
        tout.backward()

        self.assertArrayClose(out.data, tout.detach().numpy())
        self.assertArrayClose(x.grad, tx.grad.numpy())

    def test_division_backward(self):
        numerator_data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)
        denominator_data = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=float)

        numerator = MyTensor(numerator_data)
        denominator = MyTensor(denominator_data)
        out = (numerator / denominator).sum()
        out.backward()

        t_numerator = torch.tensor(numerator_data, dtype=torch.float64, requires_grad=True)
        t_denominator = torch.tensor(denominator_data, dtype=torch.float64, requires_grad=True)
        tout = (t_numerator / t_denominator).sum()
        tout.backward()

        self.assertArrayClose(out.data, tout.detach().numpy())
        self.assertArrayClose(numerator.grad, t_numerator.grad.numpy())
        self.assertArrayClose(denominator.grad, t_denominator.grad.numpy())


if __name__ == "__main__":
    unittest.main()
