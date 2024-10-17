
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score, cohen_kappa_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import xgboost as xgb

# 1. 读取数据集
data = pd.read_excel(r'C:\XXX.xls', sheet_name='Landslide data')

# 2. 数据预处理
X = data.drop(columns=['L'])  # 特征列
y = data['L']  # 标签列

# 3. 划分数据集为训练集和验证集
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=123)

# 4. 定义超参数搜索空间
param_space = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 5),
    'min_child_weight': Integer(1, 10)
}

# 5. 使用Bayesian Optimization进行参数搜索和交叉验证
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
bayes_search = BayesSearchCV(xgb_model, param_space, cv=10, scoring='roc_auc', n_jobs=None, n_iter=50)
bayes_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", bayes_search.best_params_)

# 6. 手动提取交叉验证结果
cv_results = bayes_search.cv_results_
mean_test_scores = cv_results['mean_test_score']
split_test_scores = [cv_results[f'split{i}_test_score'] for i in range(10)]

# 构建DataFrame
results_df = pd.DataFrame({'Mean Test Score': mean_test_scores})
for i in range(10):
    results_df[f'Split {i} Test Score'] = split_test_scores[i]

# 7. 使用最佳参数在Validation set上进行验证
best_model = bayes_search.best_estimator_

# 8. 使用最佳参数在holdout set上进行验证
y_pred_holdout = best_model.predict(X_holdout)
y_pred_proba_holdout = best_model.predict_proba(X_holdout)[:, 1]

# 9. 评估模型性能（在holdout set上）
fpr_holdout, tpr_holdout, threshold_holdout = roc_curve(y_holdout, y_pred_proba_holdout)

auc_score_holdout = auc(fpr_holdout, tpr_holdout)
recall_holdout = recall_score(y_holdout, y_pred_holdout)
precision_holdout = precision_score(y_holdout, y_pred_holdout)
f1_holdout = f1_score(y_holdout, y_pred_holdout)
accuracy_holdout = accuracy_score(y_holdout, y_pred_holdout)
kappa_holdout = cohen_kappa_score(y_holdout, y_pred_holdout)

print("\nHoldout Set Performance:")
print("AUC:", auc_score_holdout)
print("Recall:", recall_holdout)
print("Precision:", precision_holdout)
print("F1 Score:", f1_holdout)
print("Overall Accuracy:", accuracy_holdout)
print("Kappa:", kappa_holdout)

# 10. 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr_holdout, tpr_holdout, color='green', lw=2, label='Holdout Set ROC curve (area = %0.2f)' % auc_score_holdout)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity', fontdict={'family': 'Arial', 'weight': 'heavy', 'size': 12})
plt.ylabel('Sensitivity', fontdict={'family': 'Arial', 'weight': 'heavy', 'size': 12})

plt.legend(loc="lower right")
plt.grid(False)
plt.savefig(r'XXX\XGB.tif', dpi=500)
plt.show()

# 11. 保存交叉验证结果
results_df.to_excel(r'XXX\XGB.xlsx', index=False)