import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)

def visualize_data(xs, ys):
    """可视化数据。"""
    plt.scatter(xs, ys)
    plt.show()

def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret



def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    #==========
    poly = [x**i for i in range(1, feature_num+1)]
    ret = np.concatenate(poly, axis=1)
    #==========
    return ret

def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    #==========
    gauss = [np.exp(-x*x/(2*i)) for i in range(1, feature_num)]
    ret = np.concatenate([x]+gauss, axis=1)
    #==========
    
    return ret

def main(x_train, y_train):
    """
    训练模型，并返回从x到y的映射。
    
    """
    # basis_func = identity_basis
    basis_func = multinomial_basis
    # basis_func = gaussian_basis
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi = np.concatenate([phi0, phi1], axis=1)
    
    
    #==========
    # 最小二乘
    # w = np.linalg.inv(phi.T @ phi) @ phi.T @ y_train
    # 梯度下降
    # Standardize polynomial features so x^2 / x^3 terms do not blow up updates.
    feature_mean = phi[:, 1:].mean(axis=0)
    feature_std = phi[:, 1:].std(axis=0)
    feature_std[feature_std == 0] = 1.0

    phi_norm = phi.copy()
    phi_norm[:, 1:] = (phi_norm[:, 1:] - feature_mean) / feature_std

    rate = 0.01
    epochs = 1000000
    w = np.zeros(phi_norm.shape[1])
    n_samples = phi_norm.shape[0]
    for _ in range(epochs):
        grad = phi_norm.T @ (phi_norm @ w - y_train) / n_samples
        w -= rate * grad
        if _ % 1000 == 0:
            print(grad)
    #==========
    
    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        phi[:, 1:] = (phi[:, 1:] - feature_mean) / feature_std
        y = np.dot(phi, w)
        return y

    return f

def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

# 程序主入口（建议不要改动以下函数的接口）
if __name__ == '__main__':
    train_file = 'chap2_linear_regression/train.txt'
    test_file = 'chap2_linear_regression/test.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))
    
    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    #显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
#     plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()
