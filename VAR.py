import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

# 读取数据
df = pd.read_csv(r"C:\Users\千骑卷平冈\Desktop\2023.12.1\1.csv", index_col='Date', parse_dates=True)
nobs = 4  # 根据需要调整
df_train, df_test = df[0:-nobs], df[-nobs:]

# 拟合VAR模型
model = VAR(df_train)
model_fit = model.fit()

# 进行预测
lag_order = model_fit.k_ar
prediction = model_fit.forecast(df_train.values[-lag_order:], steps=nobs)
print(prediction)