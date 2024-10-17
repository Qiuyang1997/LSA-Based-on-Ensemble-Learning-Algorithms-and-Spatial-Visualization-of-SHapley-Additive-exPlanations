import numpy as np
np.float = float
import pandas as pd
import rasterio
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 读取数据集
data = pd.read_excel(r'XXXX.xls', sheet_name='Landslide data')

# 2. 数据预处理
X = data.drop(columns=['L'])  # 特征列
y = data['L']  # 标签列

# 4. 定义超参数搜索空间

param_rfc = {
    'n_estimators': 530,
    'max_depth': 28,
    'max_features': 'sqrt',
    'min_samples_split': 4,
    'min_samples_leaf': 5,
    'criterion': 'entropy'
}

param_gbdt = {
    'n_estimators': 506,
    'max_depth': 10,
    'learning_rate': 0.01,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'subsample': 0.7
}

param_xgb = {
    'n_estimators': 586,
    'max_depth': 10,
    'learning_rate': 0.01,
    'subsample': 0.7234778834233403,
    'colsample_bytree': 0.7006831321656208,
    'gamma': 2.1129915818201335,
    'min_child_weight': 1
}

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
rfc = RandomForestClassifier(**param_rfc)
gbdt = GradientBoostingClassifier(**param_gbdt)
xgb = xgb.XGBClassifier(**param_xgb)
lgbm = lgb.LGBMClassifier(**param_lgbm)

# 6. 训练模型
#rfc.fit(X, y)
gbdt.fit(X, y)
xgb.fit(X, y)
lgbm.fit(X, y)

# 栅格文件路径列表
input_raster_filepaths = ["G:XXX\Slope gradient.tif",
                          "G:XXX\Aspect of slope.tif",
                          "G:XXX\Lithology.tif",
                          "G:XXX\Distance form faults.tif",
                          "G:XX\Distance form anticlines.tif",
                          "G:XXX\Distance form synclines.tif",
                          "G:XXXX\Distance from rivers.tif",
                          "G:XXX\Distance from mineral sites.tif",
                          "G:XXX\Distance from transport lines.tif",
                          "G:XXXX\Land use types.tif"]

# 读取栅格数据和元数据信息
raster_data = []
profiles = []
for file_path in input_raster_filepaths:
    with rasterio.open (file_path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr<-3e+38]=np.nan###删除无效值
        raster_data.append(arr)
        profiles.append(src.profile)
        rows, clos = src.shape

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

# 创建一个带有特征名称的数据框
# 去掉后缀并创建一个带有特征名称的数据框
feature_names = [fp.split("\\")[-1].split('.')[0] for fp in input_raster_filepaths]
reshaped_raster_df = pd.DataFrame(reshaped_raster, columns=feature_names)
# 获取含有NaN值的索引
nan_indices = reshaped_raster_df.isna().any(axis=1)

# 移除包含缺失值的行
reshaped_raster_df = reshaped_raster_df.dropna()


# 进行预测
predictions1 = rfc.predict_proba(reshaped_raster_df)[:, 1]

# 将预测结果转换为栅格形状
predicted_raster1 = predictions1.reshape((rows, cols))

# 输出栅格文件路径
output_filepath1 = r'XXXX/RF.tif'
profile.update(dtype=rasterio.float32, count=1)

# 创建输出栅格文件并写入预测结果
with rasterio.open(output_filepath1, 'w', **profile) as dst:
    dst.write(predicted_raster1, 1)

print("预测结果已保存至: ", output_filepath1)


# 进行预测
predictions2 = gbdt.predict_proba(reshaped_raster_df)[:, 1]

# 将预测结果转换为栅格形状
# 将预测结果重新映射回原始栅格形状
predicted_raster2 = np.full((rows * cols), np.nan, dtype=np.float32)
predicted_raster2[~nan_indices] = predictions2
predicted_raster2 = predicted_raster2.reshape((rows, cols))

# 输出栅格文件路径
output_filepath2 = r'XXXXGBDT.tif'
profile.update(dtype=rasterio.float32, count=1)

# 创建输出栅格文件并写入预测结果
with rasterio.open(output_filepath2, 'w', **profile) as dst:
    dst.write(predicted_raster2, 1)

print("预测结果已保存至: ", output_filepath2)


# 进行预测
predictions3 = xgb.predict_proba(reshaped_raster_df)[:, 1]

# 将预测结果转换为栅格形状
predicted_raster3 = np.full((rows * cols), np.nan, dtype=np.float32)
predicted_raster3[~nan_indices] = predictions3
predicted_raster3 = predicted_raster3.reshape((rows, cols))

# 输出栅格文件路径
output_filepath3 = r'XXXX\XGB.tif'
profile.update(dtype=rasterio.float32, count=1)

# 创建输出栅格文件并写入预测结果
with rasterio.open(output_filepath3, 'w', **profile) as dst:
    dst.write(predicted_raster3, 1)

print("预测结果已保存至: ", output_filepath3)



# 进行预测
predictions4 = gbdt.predict_proba(reshaped_raster_df)[:, 1]

# 将预测结果转换为栅格形状

predicted_raster4 = np.full((rows * cols), np.nan, dtype=np.float32)
predicted_raster4[~nan_indices] = predictions4
predicted_raster4 = predicted_raster4.reshape((rows, cols))
# 输出栅格文件路径
output_filepath4 = r'XXXX\LGBM.tif'
profile.update(dtype=rasterio.float32, count=1)

# 创建输出栅格文件并写入预测结果
with rasterio.open(output_filepath4, 'w', **profile) as dst:
    dst.write(predicted_raster4, 1)

print("预测结果已保存至: ", output_filepath4)
