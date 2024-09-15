import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义微分方程
def ecological_model(t, y, energy_input, population, traditional_vehicles, electric_vehicles):
    k_energy = 0.1  # 能源排放系数
    beta_energy = 0.05  # 能源吸收速率
    k_population = 0.002  # 人口对能源的消耗系数
    k_traditional = 0.03  # 传统燃油汽车的CO2排放系数
    k_electric = 0.01  # 新能源汽车的CO2排放系数
    dydt = k_energy * energy_input - beta_energy * y[0] - k_population * population - k_traditional * traditional_vehicles - k_electric * electric_vehicles  # CO2浓度的变化率
    return dydt

# 模拟能源电动化对CO2浓度的影响，考虑人口、汽车保有量和能源消耗
energy_input = 0.5  # 能源输入，假设电动化会减少能源排放
initial_population = 1000000  # 初始人口
initial_co2_concentration = 300  # 初始CO2浓度
initial_traditional_vehicles = 700000  # 初始传统燃油汽车保有量
initial_electric_vehicles = 30000  # 初始新能源汽车保有量
y0 = [initial_co2_concentration]  # 初始状态
t_span = (0, 100)  # 时间范围
t_eval = np.linspace(0, 100, 100)  # 时间点

# 求解微分方程
sol = solve_ivp(ecological_model, t_span, y0, t_eval=t_eval, args=(energy_input, initial_population, initial_traditional_vehicles, initial_electric_vehicles))

# 可视化结果
plt.plot(sol.t, sol.y[0], label='CO2 Concentration')
plt.xlabel('Time')
plt.ylabel('CO2 Concentration')
plt.title('Impact of Electrification on CO2 Concentration with Population and Vehicle Ownership')
plt.legend()
plt.show()
