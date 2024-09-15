import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


data_train_set = pd.read_csv(r"C:\Users\千骑卷平冈\Desktop\2023.12.1\副本Problem  2 Result.csv")
data_train_set.head()


d = data_train_set.corr()
display(d)

plt.subplots(figsize = (12,12))
sns.heatmap(d,annot = True,vmax = 1,square = True,cmap = "Blues")
plt.title('Correlation coefficient map for Problem 2')
plt.show()
