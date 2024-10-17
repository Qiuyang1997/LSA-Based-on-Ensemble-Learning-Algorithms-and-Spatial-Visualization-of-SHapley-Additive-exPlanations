import shap
import matplotlib as mpl
import numpy as np
np.float = float
import pandas as pd
import rasterio
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


# 栅格文件路径列表
input_raster_filepaths = ["XXXX\Slope gradient.tif",
                          "XXXX\Aspect of slope.tif",
                          "XXXX\Lithology.tif",
                          "XXXX\Distance form faults.tif",
                          "XXXX\Distance form anticlines.tif",
                          "XXXX\Distance form synclines.tif",
                          "XXXX\Distance from rivers.tif",
                          "XXXXX\Distance from mineral sites.tif",
                          "XXXX\Distance from transport lines.tif",
                          "XXXX\Land use types.tif"]

# 读取栅格数据和元数据信息
raster_data = []
profiles = []
for file_path in input_raster_filepaths:
    with rasterio.open(file_path) as src:
        raster_data.append(src.read(1).astype(np.float32))
        profiles.append(src.profile)


# 使用第一个输入栅格的坐标转换信息
profile = profiles[0]

# 将坐标转换信息应用于输出栅格文件
transform = profile["transform"]

# 将栅格数据堆叠为多维数组
stacked_raster = np.stack(raster_data, axis=-1)

# 获取栅格数据的行数、列数和波段数
rows, cols, num_bands = stacked_raster.shape

# 将栅格数据转换为2D数组形式
reshaped_raster = stacked_raster.reshape((rows * cols, num_bands))

# 5. 定义模型

lgbm = lgb.LGBMClassifier(**param_lgbm)
lgbm.fit(X, y)
#计算SHAP
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(reshaped_raster)
shap_values_1 =shap_values[1]


# 创建输出栅格文件并写入SHAP值栅格图
for i, input_raster_filepath in enumerate(input_raster_filepaths):
    output_filepath = r'XXXX/SHAP' + input_raster_filepath.split('\\')[-1].split('.')[0] + 'SHAP visualization.tif'
    shap_raster = shap_values_1[:, i].reshape((rows, cols))  # 获取特定栅格的SHAP值栅格
    profile.update(count=1, dtype=rasterio.float32)
    with rasterio.open(output_filepath, 'w', **profile) as dst:
        dst.write(shap_raster, 1)
    print("栅格文件 {} 对应的 SHAP 值栅格已保存至: {}".format(input_raster_filepath, output_filepath))