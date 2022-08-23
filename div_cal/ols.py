import numpy as np

# 样本在列上
# Y = np.array([[1, 3, 5, 10, 4],
#               [2, 6, 10, 20, 8],
#               [3, 9, 15, 30, 12],
#               ])
observe_j = 3
pre_k = 3
# 直接套公式 Beta=(X'Y)/(X'X) Y=BetaX
X = np.array([[1] * observe_j, range(observe_j)]).T
X_PRE = np.array([[1] * pre_k, range(observe_j, observe_j + pre_k)]).T
# 第一行不要变
# 第二行每一列是每个样本要预测的值
# X_PRE = np.array([[1, 4, ],
#                   [1, 5, ],
#                   [1, 6, ]])

# 样本在行上
Y = np.array([[1, 2, 3],
              [3, 6, 9],
              [5, 10, 15],
              ]).T

# X = np.array([1, 2, 3])
# X = np.hstack([X.reshape(-1, 1), np.ones([X.shape[0], 1])])
BETA = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

PRE = X_PRE.dot(BETA).T

#
# x = np.array([[1, 1],
#               [2, 1],
#               [3, 1]])
# # x = np.hstack([x.reshape(-1, 1), np.ones([x.shape[0], 1])])
# y = np.array([2, 3, 4])
# beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# print(np.cov(Y, X))
# def getOLSBeta(x, y):
#     x = np.hstack([x.reshape(-1, 1), np.ones([x.shape[0], 1])])
#     x_pre = np.hstack([x_pre.reshape(-1, 1), np.ones([x_pre.shape[0], 1])])
# beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# pre=x_pre.dot(beta)
# return beta

# getOLSBeta(X, Y, X_PRE)
