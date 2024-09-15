# 导入必要的库
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 提供的训练数据
y_train = np.array([0.8, 1.2, 1.7, 7.5, 33.1, 50.7, 77.7, 125.6, 120.6])
X_train = np.array([5, 7, 9, 12, 15, 18, 21, 24, 27]).reshape(-1, 1)

# 提供的测试数据
X_test = np.array([30, 33, 36]).reshape(-1, 1)
y_test_true = np.array([136, 351, 688.7])

# 创建决策树回归模型并拟合训练数据
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(X_train, y_train)

# 对测试数据进行预测
y_pred = regr.predict(X_test)

# 计算MAE
mae = mean_absolute_error(y_test_true, y_pred)
print(f'MAE: {mae:.2f}')

# 计算MSE
mse = mean_squared_error(y_test_true, y_pred)
print(f'MSE: {mse:.2f}')

# 计算R²
r2 = r2_score(y_test_true, y_pred)
print(f'R²: {r2:.2f}')

# 绘制结果
plt.scatter(X_train, y_train, s=50, edgecolor="black", c="darkorange", label="Training data")
plt.scatter(X_test, y_test_true, s=50, edgecolor="black", c="blue", label="True test data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="Predicted test data", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


