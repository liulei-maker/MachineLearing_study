# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # 计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征


# def meanX(data):
#     return np.mean(data, axis=0)  # axis=0表示按照列来求均值，如果输入list,则axis=1

# # 计算类内离散度矩阵子项si


# def compute_si(xi):
#     n = xi.shape[0]
#     ui = meanX(xi)
#     si = 0
#     for i in range(0, n):
#         si = si + (xi[i, :] - ui).T * (xi[i, :] - ui)
#     return si

# # 计算类间离散度矩阵Sb


# def compute_Sb(x1, x2):
#     dataX = np.vstack((x1, x2))  # 合并样本
#     print("dataX:", dataX)
#     # 计算均值
#     u1 = meanX(x1)
#     u2 = meanX(x2)
#     u = meanX(dataX)  # 所有样本的均值
#     Sb = (u-u1).T * (u-u1) + (u-u2).T * (u-u2)
#     return Sb


# def LDA(x1, x2):
#     # 计算类内离散度矩阵Sw
#     s1 = compute_si(x1)
#     s2 = compute_si(x2)
#     # Sw=(n1*s1+n2*s2)/(n1+n2)
#     Sw = s1 + s2

#     # 计算类间离散度矩阵Sb
#     # Sb=(n1*(m-m1).T*(m-m1)+n2*(m-m2).T*(m-m2))/(n1+n2)
#     Sb = compute_Sb(x1, x2)

#     # 求最大特征值对应的特征向量
#     eig_value, vec = np.linalg.eig(np.mat(Sw).I*Sb)  # 特征值和特征向量
#     index_vec = np.argsort(-eig_value)  # 对eig_value从大到小排序，返回索引
#     eig_index = index_vec[:1]  # 取出最大的特征值的索引
#     w = vec[:, eig_index]  # 取出最大的特征值对应的特征向量
#     return w


# def createDataSet():
#     X1 = np.mat(np.random.random((8, 2)) * 5 + 15)  # 类别A
#     X2 = np.mat(np.random.random((8, 2)) * 5 + 2)  # 类别B
#     return X1, X2


# x1, x2 = createDataSet()

# print(x1)
# print("\r\n")
# print(x2)
# print("\r\n")

# # LDA训练
# w = LDA(x1, x2)
# print("w:", w)

# # 编写一个绘图函数


# def plotFig(group):
#     fig = plt.figure()
#     plt.ylim(0, 30)
#     plt.xlim(0, 30)
#     ax = fig.add_subplot(111)
#     ax.scatter(group[0, :].tolist(), group[1, :].tolist())
#     plt.show()


# plotFig(np.hstack((x1.T, x2.T)))

# test2 = np.mat([2, 8])
# g = np.dot(w.T, test2.T - 0.5 * (meanX(x1)-meanX(x2)).T)
# print("Output: ", g)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from mpl_toolkits.mplot3d import Axes3D


def LDA(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

    len1 = len(X1)
    len2 = len(X2)

    mju1 = np.mean(X1, axis=0)  # 求中心点
    mju2 = np.mean(X2, axis=0)

    cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
    cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
    Sw = cov1 + cov2

    w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1), 1)))  # 计算w
    X1_new = func(X1, w)*w[0][0]/(w.T*w)
    X2_new = func(X2, w)*w[0][0]/(w.T*w)
    # y1_new = [1 for i in range(len1)]
    # y2_new = [2 for i in range(len2)]
    y1_new = func(X1, w)*w[1][0]/(w.T*w)
    y2_new = func(X2, w)*w[1][0]/(w.T*w)
    return X1_new, X2_new, y1_new, y2_new, w


def func(x, w):
    return np.dot((x), w)


if '__main__' == __name__:
    # X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2,
    #                            n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    # print(y.shape)
    # print(X, "\r\n", y)
    # X = np.mat(np.random.random((8, 2)) * 5 + 15)  # 类别A
    # y = np.mat(np.random.random((8, 2)) * 5 + 8)  # 类别B

    # print("X:", X)

    # print("y:", y)
    X = np.array([[17, 16],
                  [16, 19],
                  [17, 19],
                  [17, 19],
                  [19, 18],
                  [19, 18],
                  [16, 18],
                  [17, 17],
                  [9,  10],
                  [12, 9],
                  [9,  10],
                  [11, 9],
                  [11, 8],
                  [10, 11],
                  [12, 11],
                  [12, 11]])
    y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    X1_new, X2_new, y1_new, y2_new, w = LDA(X, y)
    print("w:", w)
    b = [0]*2
    x1 = np.arange(0, 20, 0.1)
    x2 = np.array((w[1][0] * x1) / (w[0][0]))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    # plt.plot(x1, x2[0, :],)
    plt.plot(X1_new, y1_new, 'b*')
    plt.plot(X2_new, y2_new, 'ro')
    plt.plot(x1, x2[0, :],)
    plt.show()
