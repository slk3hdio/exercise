import numpy as np

def ensure_array(data):
    if isinstance(data, np.ndarray):
        return data.astype(float)
    return np.array(data, dtype=float)

def unbroadcast(grad, shape):
    """
    把广播后的梯度压回原始 shape
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class MyTensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = ensure_array(data)
        self.grad = np.zeros_like(self.data, dtype=float)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    # -------- 基本运算 --------
    def __add__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        out = MyTensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        out = MyTensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        return self * other.pow(-1)

    def pow(self, power):
        out = MyTensor(self.data ** power, (self,), f'pow({power})')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        return self.pow(power)

    # -------- 矩阵运算 --------
    def matmul(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        out = MyTensor(self.data @ other.data, (self, other), 'matmul')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    @property
    def T(self):
        out = MyTensor(self.data.T, (self,), 'transpose')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    # -------- 常用函数 --------
    def relu(self):
        out = MyTensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = MyTensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = MyTensor(self.data.mean(), (self,), 'mean')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size

        out._backward = _backward
        return out

    # -------- 反向传播 --------
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=float)

        for node in reversed(topo):
            node._backward()