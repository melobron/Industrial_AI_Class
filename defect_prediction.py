import copy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('autoparts.csv')
data = data.loc[data['prod_no'] == '90784-76001']
row, col = data.shape
print(data.columns)

# Correlation
# sns.heatmap(new_data.corr())
# plt.show()

# Boxplot
# fig, ax = plt.subplots()
# ax.boxplot(new_data.loc[new_data.columns[0]])
# plt.title('')
# plt.xticks([1,2,3],[category[0],category[1],category[2]])
# plt.show()

# Binary
new_data = copy.deepcopy(data)
new_data.insert(col, 'failure', 1)
new_data['c_thickness'].mask(20 <= new_data['c_thickness'] <= 32, 0, inplace=True)
print(new_data.head())


X, y = new_data.iloc[:, :-1], new_data.iloc[:, -1]
train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.2, shuffle=True)


