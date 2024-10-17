import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
np.float = float
import pandas as pd

import lightgbm as lgb
# 设置全局字体样式和大小
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'bold'
data = pd.read_excel(r'XXXXX.xls', sheet_name='Landslide data')

# 2. 数据预处理
X = data.drop(columns=['L'])  # 特征列
y = data['L']  # 标签列


# 设置参数空间

param_lgbm = {
    'num_leaves': 44,
    'n_estimators': 729,
    'learning_rate': 0.001806314690673868,
    'max_depth': 8,
    'min_data_in_leaf': 10,
    'feature_fraction': 0.6199874927995336,
    'bagging_fraction': 0.6905879641995619,
    'bagging_freq': 5,
    'min_gain_to_split': 0.1
}

# 5. 定义模型

lgbm = lgb.LGBMClassifier(**param_lgbm)
lgbm.fit(X, y)
#计算SHAP
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X)
#特征交互图
# 设置全局字体样式和大小
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'bold'
shap.summary_plot(shap_values, X, show=False)
plt.savefig(r'XXXXX\Native SHAP.tif', dpi=500)
# 选择shap_values列表中的第一个数组
shap_values_first = shap_values[1]

shap.dependence_plot('Slope gradient', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('SG', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of SG.tif', dpi=500)

shap.dependence_plot('Aspect of slope', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('AS', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of AS.tif', dpi=500)

shap.dependence_plot('Lithology', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('LIT', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of Lithology.tif', dpi=500)

shap.dependence_plot('Distance form faults', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFF', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFF.tif', dpi=500)

shap.dependence_plot('Distance form anticlines', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFA', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFA.tif', dpi=500)

shap.dependence_plot('Distance form synclines', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFS', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of DFS.tif', dpi=500)

shap.dependence_plot('Distance from rivers', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFR', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFR.tif', dpi=500)

shap.dependence_plot('Distance from mineral sites', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFM', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFM.tif', dpi=500)

shap.dependence_plot('Distance from transport lines', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFT', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of DFT.tif', dpi=500)

shap.dependence_plot('Land use types', shap_values_first, X, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('LU', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of LU.tif', dpi=500)



plt.show()

