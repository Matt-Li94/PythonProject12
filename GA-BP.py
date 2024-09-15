import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 原始数据
data = np.array([10.9,7.7,23.4,60.2,65.8,64.8,62.8,52.4,48.6,60.5])
data = data.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# 创建神经网络
input_data = data_normalized[:-1]
target = data_normalized[1:]
net = nl.net.newff([[0, 1]], [10, 1])

# 配置训练方法
net.trainf = nl.train.train_gd

# 训练神经网络
error = net.train(input_data, target, epochs=1000, show=100, goal=0.02)

# 预测
predicted = net.sim(input_data)

# 反归一化
predicted = scaler.inverse_transform(predicted)
target = scaler.inverse_transform(target)

# # 可视化结果
# plt.plot(data, label='Original Data')
# plt.plot(np.arange(1, len(data)), predicted, label='Predicted Data', linestyle='--')
# plt.legend()
# plt.show()

# 最后一个训练数据点
last_data_point = data_normalized[-1]

# 预测未来的数据点
future_steps = 10  # 你可以根据需要设定预测未来的步数
future_data = []

for _ in range(future_steps):
    # 使用神经网络进行预测
    predicted_value = net.sim([last_data_point])

    # 将预测值添加到结果中
    future_data.append(predicted_value[0, 0])

    # 更新最后一个数据点，用于下一步预测
    last_data_point = np.array([predicted_value[0, 0]])

# 反归一化
future_data = scaler.inverse_transform(np.array(future_data).reshape(-1, 1))

print(future_data);
# 可视化结果
# 可视化结果
plt.plot(data, label='Original Data')
plt.plot(np.arange(1, len(data)+1), scaler.inverse_transform(predicted), label='Predicted Data', linestyle='--')
plt.plot(np.arange(len(data)+2, len(data)+future_steps+1), future_data, label='Predicted Future Data', linestyle='--')
plt.legend()
plt.show()

