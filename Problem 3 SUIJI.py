import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 已知的太阳黑子数数据
data = np.array([1259060, 1432736, 1004423, 1092296, 1180664, 1844908, 2109346, 2247640, 1502317, 2331525, 2232118, 1956669, 946727])

# 创建特征和标签数据集
X = np.arange(1, 14).reshape(-1, 1)  # 特征数据集（周期数）
Y = data.reshape(-1, 1)  # 标签数据集（太阳黑子数）

# 创建随机森林回归模型并拟合数据
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)

# 进行预测
X_pred = np.arange(1, 14).reshape(-1, 1)  # 需要预测的特征数据集
Y_pred = model.predict(X_pred)  # 预测的标签数据集（太阳黑子数）

# 输出预测值
print("预测值：", Y_pred.flatten())