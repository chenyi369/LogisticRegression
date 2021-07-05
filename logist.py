import pandas as pd
import numpy as np

def read_xlsx(path):
    data = pd.read_excel(path)
    print(data)
    return data

def MinMaxScaler(data):
    col = data.shape[1]
    for i in range(0, col-1):
        arr = data.iloc[:, i]
        arr = np.array(arr)
        min = np.min(arr)
        max = np.max(arr)
        arr = (arr-min)/(max-min)
        data.iloc[:, i] = arr
    return data

def train_test_split(data, test_size=0.2, random_state=None):
    col = data.shape[1]
    x = data.iloc[:, 0:col-1]
    y = data.iloc[:, -1]
    x = np.array(x)
    y = np.array(y)
    # 设置随机种子，当随机种子非空时，将锁定随机数
    if random_state:
        np.random.seed(random_state)
        # 将样本集的索引值进行随机打乱
        # permutation随机生成0-len(data)随机序列
    shuffle_indexs = np.random.permutation(len(x))
    # 提取位于样本集中20%的那个索引值
    test_size = int(len(x) * test_size)
    # 将随机打乱的20%的索引值赋值给测试索引
    test_indexs = shuffle_indexs[:test_size]
    # 将随机打乱的80%的索引值赋值给训练索引
    train_indexs = shuffle_indexs[test_size:]
    # 根据索引提取训练集和测试集
    x_train = x[train_indexs]
    y_train = y[train_indexs]
    x_test = x[test_indexs]
    y_test = y[test_indexs]
    # 将切分好的数据集返回出去
    # print(y_train)
    return x_train, x_test, y_train, y_test

def sigmoid(x, theta):
    # 线性回归模型，中间模型,np.dot为向量点积
    z = np.dot(x, theta)
    h = 1/(1 + np.exp(-z))
    return h

def costFunction(h, y):
    m = len(h)
    J = -1/m * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return J

def gradeDesc(x,y,alpha=0.01,iter_num=2000):
    m = x.shape[0]
    n = x.shape[1]
    xMatrix = np.mat(x)
    yMatrix = np.mat(y).transpose()
    J_history = np.zeros(iter_num)   # 初始化J_history, np.zero生成1行iter_num列都是0的矩阵
    theta = np.ones((n, 1))    # 初始化theta, np.zero生成n行1列都是0的矩阵
    # 执行梯度下降
    for i in range(iter_num):
        h = sigmoid(xMatrix, theta)  # sigmoid 函数
        J_history[i] = costFunction(h, y)
        theta = theta + alpha * xMatrix.transpose() * (yMatrix - h)  # 梯度
    return J_history, theta


def score(h, y):
    m = len(h)
    # 定义计数变量
    count = 0
    for i in range(m):
        if np.where(h[i] >= 0.5, 1, 0) == y[i]:
            count += 1
    accuracy = count/m
    print("Accuracy:", accuracy)
    return accuracy


if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\regression.xlsx')
    scaler_data = MinMaxScaler(data)
    x_train, x_test, y_train, y_test = train_test_split(scaler_data)
    # 调用梯度下降函数获取训练好的theta同时记录J_history
    J_history, theta = gradeDesc(x_train, y_train, alpha=0.01, iter_num=20000)
    y_pred = sigmoid(x_test, theta)
    print(score(y_pred, y_test))
