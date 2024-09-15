import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 输入数据
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
y = np.array([1259060, 1432736, 1004423, 1092296, 1180664, 1844908, 2109346, 2247640, 1502317, 2331525, 2232118, 1956669, 946727])

# 转换 x 为多项式特征矩阵
X = np.column_stack([x**i for i in range(1, 4)])  # 使用三次多项式进行拟合

# 拟合模型
model = LinearRegression()
model.fit(X, y)

# 预测
x_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
X_pred = np.column_stack([x_pred**i for i in range(1, 4)])  # 使用相同的多项式特征矩阵进行预测
y_pred = model.predict(X_pred)

# 计算均方误差和决定系数
mse = mean_squared_error(y, model.predict(X))
r2 = r2_score(y, model.predict(X))

print("均方误差 (MSE):", mse)
print("决定系数 (R^2):", r2)
print("预测结果:", y_pred)