# minist
数据说明<br>
&emsp; &emsp; 数据集为numpy格式，可用np.load加载，共四个数组，训练集x、y，测试集x、y<br>
&emsp; &emsp; 大小为x_train (60000 , 28 , 28)，t_train (60000 , 1)，x_test (10000 , 28 , 28)，t_test (10000 , 1)<br>
&emsp; &emsp; 取值范围是0-255，0-9，0-255，0-9<br>

一、数据处理<br>
&emsp; &emsp;①	对特征变量x_train、x_test进行降维。把28*28的像素转化为1*784大小的行矩阵。<br>
&emsp; &emsp;②	将标签变量t_train、t_test转化为“one-hot”模式。<br>
&emsp; &emsp;③	对特征变量x_train、x_test进行归一化处理，都除以最大值255。<br>
 <br/>
&emsp; &emsp; 大小为x_train (60000 , 784)，t_train (60000 , 10)，x_test (10000 , 784)，t_test (10000 , 10)<br>
&emsp; &emsp; 取值范围是0-1，0-9，0-1，0-9<br>
 <br/>
二、模型设定<br>
&emsp; &emsp;（一）激活函数<br>
&emsp; &emsp;&emsp; &emsp;第一层（隐藏层）<br>
&emsp; &emsp;&emsp; &emsp;X → X∙W1 + b1 → Z = sigmoid(X∙W1 + b1)<br>
&emsp; &emsp;&emsp; &emsp;隐藏层选用sigmoid函数，对X进行非线性转化，其中W1的大小为(784,50)，Z1的大小为(60000,50)。<br>

&emsp; &emsp;&emsp; &emsp;第二层（输出层）<br>
&emsp; &emsp;&emsp; &emsp;Z → Z∙W2  + b2 → Y = softmax(Z∙W2 + b2)<br>
&emsp; &emsp;&emsp; &emsp;输出层选用softmax函数，其中Y的大小为(60000,10)，Y的每一行共10个数为预测0-9的概率。<br>

&emsp; &emsp;（二）梯度下降<br>
&emsp; &emsp;&emsp; &emsp;选用Mini-batch梯度下降法，每次循环随机选取100个样本进行迭代。<br>
&emsp; &emsp;&emsp; &emsp;进行20000次梯度下降。训练集样本量为60,000，所以每 60000/100=600 次循环，可以形成一个epoch，即达到了全部样本的数量。所以共经历了 20000/600≈33 次epoch。<br>

&emsp; &emsp;（三）学习率下降策略<br>
&emsp; &emsp;&emsp; &emsp;选用了指数衰减法，学习率a=a*γ^epoch  。这里初始学习率选用0.1，初始γ选用0.99，每经历一次epoch就会下降一次学习率。

&emsp; &emsp;（四）损失函数<br>
&emsp; &emsp;&emsp; &emsp;损失函数选用交叉熵，使用L_2正则项

&emsp; &emsp;（五）反向传播<br>
&emsp; &emsp;&emsp; &emsp;通过反向传播来求解梯度
 <br/>
三、运行<br>
&emsp; &emsp;首先，加载数据x_train_npz, t_train_npz, x_test_npz, t_test_npz = load_data(path)，并对数据预处理<br>
&emsp; &emsp;其次，建立模型network = Network(input_size, hidden_size, output_size)<br>
&emsp; &emsp;再次，梯度下降，每次循环随机选取N个样本加入迭代、调整学习率并更新参数、计算损失函数、记录测试集的准确率<br>
&emsp; &emsp;最后，绘制学习率下降曲线、测试集准确率曲线、损失函数曲线、可视化参数<br>
 <br/>
 &emsp; &emsp;function.py储存了需要用的函数，model.py是设定的模型，mnist.npz是数据集
