import numpy as np
import matplotlib.pyplot as plt
import time



from MyTensor import MyTensor


def target_function1(x):
    return (
        0.8 * np.maximum(0.0, x + 1.2)
        - 1.4 * np.maximum(0.0, x - 0.3)
        - 0.2
    )

def target_function2(x):
    return np.log(x**2+x+1/2)

def target_function3(x):
    return np.sin(0.7*x**2+x)


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


def build_dataset(num_points=300, target_function=target_function1):
    xs = np.linspace(-2.0, 2.0, num_points).reshape(-1, 1)
    ys = target_function(xs)
    return xs, ys


def split_dataset(xs, ys, train_ratio=0.80, seed=0):
    # 设置随机种子以保证可复现性
    rng = np.random.default_rng(seed)
    n = len(xs)
    indices = np.arange(n)
    rng.shuffle(indices)
    split_idx = int(n * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_indices.sort()
    test_indices.sort()
    return xs[train_indices], ys[train_indices], xs[test_indices], ys[test_indices]


def train_network(epochs=5000, lr=0.03, hidden_dim=48, seed=0, target_function=target_function1):
    xs, ys = build_dataset(target_function=target_function)
    x_train, y_train, x_test, y_test = split_dataset(xs, ys)
    x = MyTensor(x_train)
    y = MyTensor(y_train)
    x_eval = MyTensor(x_test)
    y_eval = MyTensor(y_test)

    net = TwoLayerReLUNet(hidden_dim=hidden_dim, seed=int(time.time()))
    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        pred = net(x)
        loss = mse_loss(pred, y)
        test_pred = net(x_eval)
        test_loss = mse_loss(test_pred, y_eval)

        train_loss_history.append(loss.data.item())
        test_loss_history.append(test_loss.data.item())

        net.zero_grad()
        loss.backward()
        net.step(lr)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"epoch={epoch:5d}, "
                f"train_loss={loss.data.item():.6f}, "
                f"test_loss={test_loss.data.item():.6f}"
            )

    train_pred = net(x).data
    test_pred = net(x_eval).data
    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)
    max_abs_error = np.max(np.abs(test_pred - y_test))
    return (
        net,
        x_train,
        y_train,
        x_test,
        y_test,
        train_pred,
        test_pred,
        train_mse,
        test_mse,
        max_abs_error,
        train_loss_history,
        test_loss_history,
    )


def plot_results(
    x_train,
    y_train,
    x_test,
    y_test,
    train_pred,
    test_pred,
    train_loss_history,
    test_loss_history,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(x_train[:, 0], y_train[:, 0], label="Train Target", linewidth=2)
    axes[0].plot(
        x_train[:, 0],
        train_pred[:, 0],
        label="Train Prediction",
        linewidth=2,
        linestyle="--",
    )
    axes[0].scatter(x_test[:, 0], y_test[:, 0], label="Test Target")
    axes[0].scatter(
        x_test[:, 0],
        test_pred[:, 0],
        label="Test Prediction",
    )
    # axes[0].scatter(
    #     x_train[:, 0],
    #     y_train[:, 0],
    #     s=14,
    #     alpha=0.35,
    #     label="Training Samples",
    # )
    # axes[0].scatter(
    #     x_test[:, 0],
    #     y_test[:, 0],
    #     s=18,
    #     alpha=0.5,
    #     label="Test Samples",
    # )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Two-Layer ReLU Fit")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_loss_history, label="Train Loss", linewidth=2)
    axes[1].plot(test_loss_history, label="Test Loss", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("Training Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def train_and_plot(target_function):
    (
        _,
        x_train,
        y_train,
        x_test,
        y_test,
        train_pred,
        test_pred,
        train_mse,
        test_mse,
        max_err,
        train_loss_history,
        test_loss_history,
    ) = train_network(target_function=target_function)
    print(f"train mse={train_mse:.6f}")
    print(f"test mse={test_mse:.6f}")
    print(f"test max abs error={max_err:.6f}")

    # Print a few sample predictions for quick verification.
    sample_ids = [0, len(x_test) // 4, len(x_test) // 2, 3 * len(x_test) // 4, len(x_test) - 1]
    for idx in sample_ids:
        print(
            f"test x={x_test[idx, 0]: .3f}, "
            f"target={y_test[idx, 0]: .3f}, "
            f"pred={test_pred[idx, 0]: .3f}"
        )

    plot_results(
        x_train,
        y_train,
        x_test,
        y_test,
        train_pred,
        test_pred,
        train_loss_history,
        test_loss_history,
    )

train_and_plot(target_function1)
train_and_plot(target_function2)
train_and_plot(target_function3)
