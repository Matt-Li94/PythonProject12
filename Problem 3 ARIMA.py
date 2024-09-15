import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 输入已知的太阳黑子数数据
data = [1259060, 1432736, 1004423, 1092296, 1180664, 1844908, 2109346, 2247640, 1502317, 2331525, 2232118, 1956669, 946727]

# 将数据转换为时间序列
data_ts = pd.Series(data)

# 拟合ARIMA模型
model = ARIMA(data_ts, order=(1, 0, 0))  # 这里使用了ARIMA(p, d, q)的参数，可以根据需要进行调整

# 拟合模型
model_fit = model.fit()

# 进行预测
predictions = model_fit.predict(start=0, end=12)  # 预测未来3个数据


# 输出预测值
print("预测值：",predictions)