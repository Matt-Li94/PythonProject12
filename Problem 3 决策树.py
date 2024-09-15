import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 已知的x和y值
x_known = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
y_known = np.array([1259060, 1432736, 1004423, 1092296, 1180664, 1844908, 2109346, 2247640, 1502317, 2331525, 2232118, 1956669, 946727])


# 将x_known转换为二维数组形式
x_known = x_known.reshape(-1, 1)

# 创建决策树回归模型
model = DecisionTreeRegressor()

# 拟合模型
model.fit(x_known, y_known)

# 预测x=11和12时的y值
x_new = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
x_new = x_new.reshape(-1, 1)
y_predicted = model.predict(x_new)

y_known_predicted = model.predict(x_known)
mse = mean_squared_error(y_known, y_known_predicted)
r2 = r2_score(y_known, y_known_predicted)

print("均方误差 (MSE):", mse)
print("决定系数 (R^2):", r2)

# 输出预测结果
for x, y in zip(x_new, y_predicted):
    print(f"当x={x}时，预测的y值为：{y}")