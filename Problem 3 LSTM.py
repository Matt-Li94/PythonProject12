import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 输入已知的太阳黑子数数据
data = np.array([1259060, 1432736, 1004423, 1092296, 1180664, 1844908, 2109346, 2247640, 1502317, 2331525, 2232118, 1956669, 946727])

# 数据预处理
data = data.reshape(len(data), 1)  # 将数据转换为二维数组，适合LSTM模型输入
data_normalized = data / np.max(data)  # 对数据进行归一化，将数据缩放到0到1之间

# 创建训练数据集
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        window = data[i:(i + look_back), 0]
        X.append(window)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5  # 定义时间窗口大小，即过去几个样本作为输入来预测下一个样本
X_train, Y_train = create_dataset(data_normalized, look_back)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=0)

# 进行预测
X_test = data_normalized[-look_back:]  # 使用最后5个样本作为输入进行预测
predictions = []
for i in range(13):  # 预测第1到第13个数据点
    X = X_test[-look_back:].reshape((1, look_back, 1))
    prediction = model.predict(X)
    predictions.append(prediction)
    X_test = np.concatenate((X_test, prediction), axis=0)

# 反归一化，将预测结果还原到原始数据范围
predictions_denormalized = np.array(predictions) * np.max(data)

# 输出预测值
print("预测值：", predictions_denormalized.flatten())