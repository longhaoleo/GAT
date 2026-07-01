"""
XGBoost 分类模块。

GeneGAT 训练完后，本项目会比较三种特征：
- raw：原始基因表达特征
- graph：GAT 池化得到的图嵌入
- hybrid：raw + graph 拼接

这个模块负责对这些特征训练同一套 XGBoost，便于做公平的消融对比。
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler

from gnns.metrics import evaluate_classification, print_metrics


# 当前实验使用的一组固定参数。想做调参时，可以传 param_dist 开启随机搜索。
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "verbosity": 0,
    "random_state": 42,
    "subsample": 1.0,
    "n_estimators": 100,
    "max_depth": 9,
    "learning_rate": 0.1,
    "gamma": 0.2,
    "colsample_bytree": 0.6,
}


def xgboost_train_or_search(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    user_params=None,
    param_dist=None,
    n_iter=20,
    random_state=42,
):
    """训练 XGBoost，或者在给定参数空间里做随机搜索。

    常用方式是只传 user_params：直接训练一版模型并在测试集评估。
    如果传入 param_dist，则会先用验证集选择最优参数，再评估测试集。
    """
    # base 里放所有实验都共用的配置；任务是二分类时使用 binary:logistic。
    base = {
        "objective": "binary:logistic" if len(np.unique(y_train)) == 2 else "multi:softprob",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": random_state,
    }

    # 项目当前默认走这条路：使用固定参数，避免调参过程干扰主流程阅读。
    if user_params and not param_dist:
        cfg = {**base, **user_params}
        model = xgb.XGBClassifier(**cfg)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        metrics = evaluate_xgb(model, X_test, y_test)
        print("使用自定义参数：", cfg)
        print_metrics("测试集评估", metrics)
        return model, cfg, metrics

    candidates = []
    if user_params:
        candidates.append(user_params)
    if param_dist:
        candidates += list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))

    # 参数搜索只看验证集分数；测试集留到最后一次性报告。
    best_score, best_cfg, best_model = 0, None, None
    for params in candidates:
        cfg = {**base, **params}
        model = xgb.XGBClassifier(**cfg)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        score = accuracy_score(y_val, model.predict(X_val))
        if score > best_score:
            best_score, best_cfg, best_model = score, cfg, model

    metrics = evaluate_xgb(best_model, X_test, y_test)
    print("最优参数：", best_cfg)
    print("验证集准确率：", best_score)
    print_metrics("测试集评估", metrics)
    return best_model, best_cfg, metrics


def evaluate_xgb(model, X_test, y_test):
    """统一计算 Accuracy、weighted F1 和二分类 AUC。"""
    y_pred = model.predict(X_test)
    try:
        y_probs = model.predict_proba(X_test)
    except Exception:
        y_probs = None
    return evaluate_classification(y_test, y_pred, y_prob=y_probs, average="weighted")
