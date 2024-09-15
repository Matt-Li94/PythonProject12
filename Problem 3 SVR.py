import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 已知的x和y值
x_known = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
y_known = np.array([1259060, 1432736, 1004423, 1092296, 1180664, 1844908, 2109346, 2247640, 1502317, 2331525, 2232118, 1956669, 946727])

# 数据标准化
scaler = StandardScaler()
x_known_scaled = scaler.fit_transform(x_known.reshape(-1, 1))
y_known_scaled = scaler.fit_transform(y_known.reshape(-1, 1))

# 参数调优
parameters = {'kernel': ('linear', 'poly', 'rbf'), 'C': [1, 10, 1]}
svr = SVR()
model = GridSearchCV(svr, parameters)
model.fit(x_known_scaled, y_known_scaled.ravel())

# 预测x=11和12时的y值
x_new = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13])
x_new_scaled = scaler.transform(x_new.reshape(-1, 1))
y_predicted_scaled = model.predict(x_new_scaled)
y_predicted = scaler.inverse_transform(y_predicted_scaled.reshape(-1, 1))

# 计算均方误差和决定系数
y_known_predicted_scaled = model.predict(x_known_scaled)
y_known_predicted = scaler.inverse_transform(y_known_predicted_scaled.reshape(-1, 1))
mse = mean_squared_error(y_known, y_known_predicted)
r2 = r2_score(y_known, y_known_predicted)

# 输出均方误差和决定系数
print("均方误差 (MSE):", mse)
print("决定系数 (R^2):", r2)

# 输出预测结果
for x, y in zip(x_new, y_predicted):
    print(f"当x={x}时，预测的y值为：{y}")