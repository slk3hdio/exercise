import numpy as np
import matplotlib.pyplot as plt

from MyTensor import MyTensor


def target_function(x):
    """
    Custom piecewise nonlinear function.
    A two-layer ReLU network should fit this family of functions well.
    """
    return (
        0.8 * np.maximum(0.0, x + 1.2)
        - 1.4 * np.maximum(0.0, x - 0.3)
        + 0.9 * np.maximum(0.0, x - 1.4)
        - 0.2
    )


class TwoLayerReLUNet:
    def __init__(self, in_dim=1, hidden_dim=48, out_dim=1, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = MyTensor(rng.normal(scale=0.5, size=(in_dim, hidden_dim)))
        self.b1 = MyTensor(np.zeros((1, hidden_dim)))
        self.w2 = MyTensor(rng.normal(scale=0.5, size=(hidden_dim, out_dim)))
        self.b2 = MyTensor(np.zeros((1, out_dim)))
        self.params = [self.w1, self.b1, self.w2, self.b2]

    def __call__(self, x):
        hidden = (x @ self.w1 + self.b1).relu()
        return hidden @ self.w2 + self.b2

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self, lr):
        for param in self.params:
            param.data -= lr * param.grad


def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).mean()


def build_dataset(num_points=200):
    xs = np.linspace(-2.0, 2.0, num_points).reshape(-1, 1)
    ys = target_function(xs)
    return xs, ys


def train_network(epochs=8000, lr=0.03, hidden_dim=32, seed=0):
    x_train, y_train = build_dataset()
    x = MyTensor(x_train)
    y = MyTensor(y_train)

    net = TwoLayerReLUNet(hidden_dim=hidden_dim, seed=seed)

    for epoch in range(epochs):
        pred = net(x)
        loss = mse_loss(pred, y)

        net.zero_grad()
        loss.backward()
        net.step(lr)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:4d}, loss={loss.data.item():.6f}")

    final_pred = net(x).data
    final_mse = np.mean((final_pred - y_train) ** 2)
    max_abs_error = np.max(np.abs(final_pred - y_train))
    return net, x_train, y_train, final_pred, final_mse, max_abs_error


def plot_fit(x_train, y_train, pred):
    plt.figure(figsize=(8, 5))
    plt.plot(x_train[:, 0], y_train[:, 0], label="Target Function", linewidth=2)
    plt.plot(
        x_train[:, 0],
        pred[:, 0],
        label="Two-Layer ReLU Prediction",
        linewidth=2,
        linestyle="--",
    )
    plt.scatter(
        x_train[:, 0],
        y_train[:, 0],
        s=14,
        alpha=0.35,
        label="Training Samples",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Two-Layer ReLU Network Fitting a Custom Function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _, x_train, y_train, pred, mse, max_err = train_network()
    print(f"final mse={mse:.6f}")
    print(f"max abs error={max_err:.6f}")

    # Print a few sample predictions for quick verification.
    sample_ids = [0, 50, 100, 150, 199]
    for idx in sample_ids:
        print(
            f"x={x_train[idx, 0]: .3f}, "
            f"target={y_train[idx, 0]: .3f}, "
            f"pred={pred[idx, 0]: .3f}"
        )

    plot_fit(x_train, y_train, pred)
