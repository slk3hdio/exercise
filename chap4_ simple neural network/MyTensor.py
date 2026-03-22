import numpy as np

class MyTensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)

        # 记录这个张量由哪些父节点计算而来
        self._prev = set(_children)
        self._op = _op

        # 当前节点的反向传播函数
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        out = MyTensor(self.data + other.data, (self, other), '+')

        def _backward():
            # out = self + other
            # d(out)/d(self) = 1
            # d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        out = MyTensor(self.data * other.data, (self, other), '*')

        def _backward():
            # out = self * other
            # d(out)/d(self) = other
            # d(out)/d(other) = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = MyTensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # 1. 先做拓扑排序，保证反向传播顺序正确
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2. 输出节点对自己的梯度设为 1
        self.grad = np.ones_like(self.data, dtype=float)

        # 3. 逆拓扑顺序反向传播
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
       