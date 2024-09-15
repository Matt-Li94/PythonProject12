# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 从CSV文件加载数据
# 假设CSV文件中有6列，前5列是自变量，最后一列是因变量
# 请替换'your_data.csv'为你的实际文件路径
# df = pd.read_csv('数据.csv')
df = pd.read_csv('数据.csv', thousands=',')


# 提取自变量和因变量
X = df.iloc[:, :-1].values  # 所有行，除了最后一列的所有列
y = df.iloc[:, -1].values   # 所有行，最后一列

# 划分训练集和测试集
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 创建随机森林回归模型并拟合训练数据
regr = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
regr.fit(X_train, y_train)

# 对测试数据进行预测
y_pred = regr.predict(X_test)

# 计算MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.2f}')

# 计算MSE
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

# 计算R²
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2:.2f}')


# 绘制结果
plt.scatter(X_train, y_train, s=50, edgecolor="black", c="darkorange", label="Training data")
plt.scatter(X_test, y_test_true, s=50, edgecolor="black", c="blue", label="True test data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="Predicted test data (Linear Regression)", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()
