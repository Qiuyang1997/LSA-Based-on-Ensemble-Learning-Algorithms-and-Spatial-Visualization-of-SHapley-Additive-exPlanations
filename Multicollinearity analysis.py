import numpy as np
import pandas as pd
# 1. 读取数据集
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
data = pd.read_excel(r'XXXXXX.xls', sheet_name='Landslide data')

# 分离特征和目标变量

X = data.drop(['L'], axis=1)
y = data['L']



vif = [variance_inflation_factor(X.values, X.columns.get_loc(i)) for i in X.columns]
print(vif)

