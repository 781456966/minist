import matplotlib.pyplot as plt
from function import *
from model import Network
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (8, 6)

# 调用load_data函数加载数据集
path = r'mnist.npz'
x_train_npz, t_train_npz, x_test_npz, t_test_npz = load_data(path)

# 对输入集进行降维并归一化
x_train = x_train_npz.reshape(60000, 28 * 28) / 255
x_test = x_test_npz.reshape(10000, 28 * 28) / 255
# 将标签转化为one-hot
t_train = one_hot_y(t_train_npz)
t_test = one_hot_y(t_test_npz)

# 查看数据集的形状
print('x_train:{}'.format(x_train.shape))
print('t_train:{}'.format(t_train.shape))
print('x_test:{}'.format(x_test.shape))
print('t_test:{}'.format(t_test.shape))

# 超参数
iters_num = 20000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
gamma = 0.99  # 学习率衰减指数

learning_rate_list = [learning_rate]
train_loss_list = []
train_acc_list = []
test_acc_list = []
per_epoch = max(train_size / batch_size, 1)  # 60000/100=600，做600次mini_batch形成一个epoch,一共经历了20000/600个epoch

# 创建网络
network = Network(input_size=784, hidden_size=50, output_size=10)

# 进行一万次mini_batch的训练
for i in range(iters_num):

    if ((i + 1) % 1000 == 0):
        print("********已经完成" + str(i + 1) + "次mini_batch...")

    # SGD，每次随机选择100个样本
    batch_num = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_num]
    t_batch = t_train[batch_num]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 指数衰减调整学习率
    if (i + 1) % per_epoch == 0:
        learning_rate = learning_rate * gamma ** ((i + 1) / per_epoch)
        learning_rate_list.append(learning_rate)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录损失函数
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 记录每次epoch的准确率
    if (i + 1) % per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
print('********运行结束')

#绘制学习率衰减曲线
plt.plot(np.arange(0,len(learning_rate_list)),learning_rate_list,label='学习率')
plt.xlabel('epochs')
plt.ylabel('学习率')
plt.title('指数衰减调整学习率(指数gamma=0.99)')
plt.legend()
plt.show()

#绘制训练集和测试集的准确率
x = np.arange(1,iters_num/per_epoch)
y1 = train_acc_list
y2 = test_acc_list

plt.plot(x,y1,label='训练集')
plt.plot(x,y2,linestyle='--',label='测试集')
plt.xlabel('epochs')
plt.ylabel('准确率')
plt.legend()
plt.show()

x = np.arange(len(train_loss_list))

plt.xlabel('mini_batch学习次数')
plt.ylabel('损失函数')
plt.title('mini-batch梯度下降法')
plt.plot(x,train_loss_list,lw=0.1)
plt.show()

#参数
w1 = network.params['W1']
b1 = network.params['b1']
w2 = network.params['W2']
b2 = network.params['b2']

print('第一层：')
print(f'**W1的系数为：{w1}')
print(f'****其shape为：{w1.shape}')
print('')
print(f'**b1的系数为：{b1}')
print(f'****其shape为：{b1.shape}')
print('')
print('第二层：')
print(f'**W2的系数为：{w2}')
print(f'****其shape为：{w2.shape}')
print('')
print(f'**b2的系数为：{b2}')
print(f'****其shape为：{b2.shape}')