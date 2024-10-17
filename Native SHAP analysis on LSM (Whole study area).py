import shap
import matplotlib as mpl
import numpy as np
np.float = float
import pandas as pd
import rasterio
import lightgbm as lgb
import random
import matplotlib.pyplot as plt
# # 设置全局字体样式和大小
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'bold'

data = pd.read_excel(r'XXXX.xls', sheet_name='Landslide data')

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


# 栅格文件路径列表
input_raster_filepaths = ["XXXX\Slope gradient.tif",
                          "XXXX\Aspect of slope.tif",
                          "XXXX\Lithology.tif",
                          "XXXX\Distance form faults.tif",
                          "XXXX\Distance form anticlines.tif",
                          "XXXX\Distance form synclines.tif",
                          "XXXX\Distance from rivers.tif",
                          "XXXX\Distance from mineral sites.tif",
                          "XXXX\Distance from transport lines.tif",
                          "XXXX\Land use types.tif"]

# 读取栅格数据
raster_data = []
for file_path in input_raster_filepaths:
    with rasterio.open(file_path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr < -3e+38] = np.nan  # 删除指定值

        raster_data.append(arr)
        # 获取栅格数据的行数和列数
        rows, cols = src.shape

# 创建新特征集Q（栅格数据）
Q = pd.DataFrame(np.stack(raster_data, axis=-1).reshape(-1, len(input_raster_filepaths)), columns=[file_path.split('\\')[-1].split('.')[0] for file_path in input_raster_filepaths])

# 5. 定义模型

lgbm = lgb.LGBMClassifier(**param_lgbm)
lgbm.fit(X, y)
#计算SHAP
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(Q)
shap_values_1 =shap_values[1]
# 输出SHAP summary plot
shap.summary_plot(shap_values, P, show=False)

plt.savefig(r'XXXX\SHAP summary.tif',dpi=500)


shap.dependence_plot('Slope gradient', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('SG', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of SG.tif', dpi=500)


shap.dependence_plot('Aspect of slope', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('AS', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of AS.tif', dpi=500)


shap.dependence_plot('Lithology', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('LIT', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of Lithology.tif', dpi=500)


shap.dependence_plot('Distance form faults', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFF', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFF.tif', dpi=500)

shap.dependence_plot('Distance form anticlines', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFA', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXX\Native SHAP of DFA.tif', dpi=500)

shap.dependence_plot('Distance form synclines', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFS', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of DFS.tif', dpi=500)

shap.dependence_plot('Distance from rivers', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFR', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFR.tif', dpi=500)

shap.dependence_plot('Distance from mineral sites', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFM', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXXX\Native SHAP of DFM.tif', dpi=500)

shap.dependence_plot('Distance from transport lines', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('DFT', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of DFT.tif', dpi=500)


shap.dependence_plot('Land use types', shap_values_1, Q, interaction_index=None,show=False)
plt.ylabel('SHAP value', fontname='Arial', fontsize=16, weight='bold')
plt.xlabel('LU', fontname='Arial', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(r'XXXX\Native SHAP of LU.tif', dpi=500)


plt.show()
