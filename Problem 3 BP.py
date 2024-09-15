import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import os

# 设置随机种子和环境变量
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 已知的x和y值
x_known = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
y_known = np.array([885684, 1178055, 1051838, 1325081, 1412367, 1841243, 2247994, 3215401, 2129870, 2255940, 1945608, 1713156, 912616])

# 创建神经网络模型
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))  # 隐藏层，10个神经元
model.add(Dense(1, activation='linear'))  # 输出层，1个神经元

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(x_known, y_known, epochs=100000, verbose=0)

# 预测x=11和12时的y值
x_new = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
y_predicted = model.predict(x_new)

# 输出均方误差和决定系数
y_known_predicted = model.predict(x_known).flatten()
mse = np.mean((y_known - y_known_predicted) ** 2)
r2 = r2_score(y_known, y_known_predicted)

# 输出均方误差和决定系数
print("均方误差 (MSE):", mse)
print("决定系数 (R^2):", r2)

# 输出预测结果
for x, y in zip(x_new, y_predicted):
    print(f"当x={x}时，预测的y值为：{y[0]}")