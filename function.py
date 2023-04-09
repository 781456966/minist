import numpy as np

def load_data(path):
    f = np.load(path) # np.load文件可以加载npz，npy格式的文件
    x_train,y_train,x_test,y_test = f['x_train'], f['y_train'], f['x_test'],f['y_test']
    f.close()
    return x_train, y_train, x_test, y_test

# 将标签转化为独热形式
def one_hot_y(arr):
    arr_zero = np.zeros((arr.size, 10))
    arr_zero[np.arange(arr.size), arr] = 1
    return arr_zero


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义softmax函数
def softmax(x):
    # 进行转置，避免内存溢出
    x = x.T
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T


# 对训练集求交叉熵损失函数,t是已知值，y是输出的概率
def cross_entropy_error(y, t):
    cost_arr = - t * np.log(y + 1e-7)  # 避免概率0的出现，无法求对数
    cost = cost_arr.sum()
    return cost


# 定义sigmoid的导数
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)