import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt

# Assuming you have performed the predictions using ARIMA, LSTM, and Random Forest models
arima_pred = [11701143,
1158100,
1352800,
1268748,
1450710,
1508837,
1794440,
2065310,
2709541,
1986647,
2070602,
1863941,
1709143

]  # ARIMA predictions
lstm_pred = [
1573981.292,
1690413.455,
1938466.652,
2150873.764,
2299902.625,
2056250.16,
1914443.928,
1748990.036,
1646303.789,
1648538.844,
1778985.621,
1887733.272,
1979470.201


]  # LSTM predictions
rf_pred = [
994531.89,
1082243.19,
1083043.31,
1254542.26,
1427557.6,
1743185.17,
2260442.96,
2869555.13,
2342203.41,
2213641.97,
2018171.59,
1746079.45,
1212552.33

]  # Random Forest predictions
xian_pred=[
1305079.308,
1371089.846,
1371089.846,
1503110.923,
1569121.462,
1635132,
1701142.538,
1767153.077,
1833163.615,
1899174.154,
1965184.692,
2031195.231,
2097205.769
]
duo_pred=[
943889.6126,
969029.6813,
1122400.008,
1363734.173,
1652765.753,
1949228.328,
2212855.476,
2403380.775,
2480537.805,
2404060.145,
2133681.372,
1629135.066,
850154.8049


]
tidu_pred=[
886402.7623,
1177247.45,
1053213.473,
1325233.518,
1412312.579,
1841416.095,
2248032.755,
3213069.117,
2131026.033,
2254110.85,
1945843.999,
1713612.697,
913331.672

]
bpshenjing_pred=[
87187.875,
130837.8281,
174487.7656,
218137.7344,
261787.7031,
305437.6875,
349087.625,
392737.5625,
436387.5313,
480037.5,
523687.4375,
567337.375,
610987.375


]
svr_pred=[
862410.1542,
862409.5247,
862408.8952,
862408.2657,
862407.6362,
862407.0067,
862406.3772,
862405.7477,
862405.1182,
862404.4887,
862403.8592,
862403.2297,
862402.6002



]

# True values
true_values = [885684, 1178055, 1051838, 1325081, 1412367, 1841243, 2247994, 3215401, 2129870, 2255940, 1945608, 1713156, 912616]
predictions = [arima_pred, lstm_pred, rf_pred,xian_pred,duo_pred,tidu_pred,bpshenjing_pred,svr_pred]


# Define the objective function: minimize prediction error
def objective(weights):
    weighted_pred = np.dot(weights, predictions)
    error = mean_squared_error(true_values, weighted_pred)
    return error

# Define the constraint: sum of weights equals 1
def constraint(weights):
    return np.sum(weights) - 1

# Initialize weight variables
initial_weights = np.array([0,0.375,0,0.125,0.125,0.125,0.125,0.125])  # Assuming equal initial weights

# Define optimization problem
optimization_problem = {'type': 'eq', 'fun': constraint}

# Solve the optimization problem
result = minimize(objective, initial_weights, constraints=optimization_problem)

# Get the optimal weights
best_weights = result.x
print("Optimal weights:", best_weights)

# 计算每个模型的预测残差
residuals_arima = np.array(true_values) - np.array(arima_pred)
residuals_lstm = np.array(true_values) - np.array(lstm_pred)
residuals_rf = np.array(true_values) - np.array(rf_pred)
residuals_xian = np.array(true_values) - np.array(xian_pred)
residuals_duo = np.array(true_values) - np.array(duo_pred)
residuals_tidu = np.array(true_values) - np.array(tidu_pred)
residuals_bpshenjing = np.array(true_values) - np.array(bpshenjing_pred)
residuals_svr = np.array(true_values) - np.array(svr_pred)

# 绘制残差的分布图
plt.figure(figsize=(10, 6))
plt.hist(residuals_arima, bins=10, alpha=0.5, label='ARIMA model')
plt.hist(residuals_lstm, bins=10, alpha=0.5, label='LSTM model')
plt.hist(residuals_rf, bins=10, alpha=0.5, label='Random Forest model')
plt.hist(residuals_xian, bins=10, alpha=0.5, label='Linear regression models')
plt.hist(residuals_duo, bins=10, alpha=0.5, label='modelsPolynomial regression ')
plt.hist(residuals_tidu, bins=10, alpha=0.5, label='Gradient boosting tree model')
plt.hist(residuals_bpshenjing, bins=10, alpha=0.5, label='BP Neural Network')
plt.hist(residuals_svr, bins=10, alpha=0.5, label='SVR')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution of Different Models')
plt.legend()
plt.show()