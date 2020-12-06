import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os, sys
os.chdir(sys.path[0])
#三维可视化函数
def ShowResult(XA, XB, YB, W1, W2, B):
    mpl.rcParams['legend.fontsize'] = 20

    font = {
        'color': 'b',
        'style': 'oblique',
        'size': 20,
        'weight': 'bold'
    }
    fig = plt.figure(figsize=(32, 24))  #参数为图片大小
    ax = fig.gca(projection='3d')  # get current axes，且坐标轴是3d的

    #输出散点图
    x = np.linspace(-1,1)
    y = np.linspace(-1,1)
    ax.set_xlabel("X", fontdict=font)
    ax.set_ylabel("Y", fontdict=font)
    ax.set_zlabel("Z", fontdict=font)   
    ax.scatter(XA, XB, YB, facecolor="gold")
    ax.legend(loc='upper right')
    #输出拟合图像
    XA, XB = np.meshgrid(x,y)
    Z = W1*XA + W2*XB + B
    ax.plot_surface(XA, XB, Z, alpha=0.5)
    plt.show()

#定义前向计算函数式，返回预测值数组Z
def returnZ(X_T,W,B):
    Z = np.dot(X_T,W) + B
    return Z

#定义损失函数，返回损失函数值
def floss(Z,Y):                            
    loss=1/2 * (Z-Y)*(Z-Y)
    return loss

#定义反向传播偏导计算函数
def LossToW1(Z,Y_T,X_TA):
    pd1=(Z-Y_T) * X_TA
    return pd1
def LossToW2(Z,Y_T,X_TB):
    pd2=(Z-Y_T) * X_TB
    return pd2


#读取文件
fname = 'mlm.csv'
csv = np.genfromtxt(fname, dtype=float, delimiter=",",skip_header=1)
date = csv[:-1]
#样本数据标签化
x_max=(date[0])[0]
y_max=(date[0])[1]
z_max=(date[0])[2]
x_min=(date[0])[0]
y_min=(date[0])[1]
z_min=(date[0])[2]

all=(date[0])[0]
for i in range(1,999):
    all=all + (date[i])[0]
x_a = all/1000

all=(date[0])[1]
for i in range(1,999):
    all=all + (date[i])[0]
y_a = all/1000

all=(date[0])[2]
for i in range(1,999):
    all=all + (date[i])[0]
z_a = all/1000

for i in range(1,999):          #max
    if (date[i])[0] > x_max:
        x_max = (date[i])[0]
    else:
        continue

for i in range(1,999):          #max
    if (date[i])[1] > y_max:
        y_max = (date[i])[1]
    else:
        continue

for i in range(1,999):          #max
    if (date[i])[2] > z_max:
        z_max = (date[i])[2]
    else:
        continue

for i in range(1,999):          #min
    if (date[i])[0] < x_min:
        x_min = (date[i])[0]
    else:
        continue

for i in range(1,999):          #min
    if (date[i])[1] < y_min:
        y_min = (date[i])[1]
    else:
        continue

for i in range(1,999):          #min
    if (date[i])[2] < z_min:
        z_min = (date[i])[2]
    else:
        continue

for i in range(0,999):
    (date[i])[0] = ((date[i])[0]-x_a) / (x_max-x_min)
for i in range(0,999):
    (date[i])[1] = ((date[i])[1]-y_a) / (y_max-y_min)
for i in range(0,999):
    (date[i])[2] = ((date[i])[2]-z_a) / (z_max-z_min)


#矩阵切割
X = date[:999,:2]
Y = date[:999,-1:]
#瞎几巴猜
w1=float(4)
w2=float(4)
b=float(2)
#设定学习率
eta=0.1
#建立W，B的矩阵
W = np.array([w1,w2])
B = np.array(b)
#循环训练
for i in range(0,999):
    X_T = X[i]
    Y_T = Y[i]
    Z = returnZ(X_T,W,B)        #前向计算
    X_TA = X_T[0]
    X_TB = X_T[1]
    loss = floss(Z,Y_T)         #损失函数计算
    pd1 = LossToW1(Z,Y_T,X_TA)
    pd2 = LossToW2(Z,Y_T,X_TB)
    pd3 = Z - Y_T
    W[0] = W[0] - eta*pd1       #反向传播
    W[1] = W[1] - eta*pd2
    B = B - eta*pd3
B = float(B)
#打印回归模型参数值
print('z={0}*x + {1}*y + {2}'.format(W[0],W[1],B))
print('损失函数值为：', loss)

#数据处理
XA = date[:999,0]
XB = date[:999,1]
YB = date[:999,2]
W1 = W[0]
W2 = W[1]

#三维可视化
'ShowResult(XA, XB, YB, W1, W2, B)'