import gnns.data
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from catboost import CatBoostClassifier


# 支持向量机分类器
def train_svm(file_path=None
            ,data=None
            , kernel='linear'
            , test_size=0.2
            , random_state=42
            ,**kwargs):
    """
    训练支持向量机（SVM）分类器。
    """
    # 获取数据
    if data is not None:
        X_train, X_test, y_train, y_test = data[0], data[1] , data[2] , data[3]
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path
                                        ,data=data,test_size=test_size, random_state=random_state)
    
    # 校验算法参数
    if kernel not in ['linear', 'rbf','poly','sigmoid']:
        raise ValueError(" 不支持的核函数，请选择 'linear'、'rbf'、'poly' 或 'sigmoid'")
    
    print(f"使用 SVM 分类器训练模型... kernel：{kernel}")

    model = SVC(kernel=kernel
                ,random_state=random_state
                ,class_weight='balanced'
                )
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# 训练 AdaBoost 分类器
def train_adaboost(file_path=None
                , data=None
                , n_estimators=50
                , learning_rate=1.0
                , algorithm='SAMME'
                , test_size=0.2
                , random_state=42
                ,**kwargs):
    """
    训练 AdaBoost 分类器。
    'SAMME': 适用于多分类，但使用的是离散 AdaBoost。
    'SAMME.R': 更常用，基于概率输出的连续 AdaBoost。兼容性问题，暂不可用
    参数:
        file_path: str，可选，数据文件路径
        data: tuple，可选，(X_train, X_test, y_train, y_test)
        n_estimators: int，弱分类器的数量
        learning_rate: float，学习率
        algorithm: str，'SAMME' 或 'SAMME.R'
        test_size: float，测试集比例
        random_state: int，随机种子
    """
    # 获取数据
    if data is not None:
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path,data=data,test_size=test_size,random_state=random_state)

    # 校验算法参数
    if algorithm not in ['SAMME', 'SAMME.R']:
        raise ValueError("不支持的算法，请选择 'SAMME' 或 'SAMME.R'")
    
    print(f"使用 AdaBoost 分类器训练模型... [n_estimators={n_estimators}, learning_rate={learning_rate}, algorithm='{algorithm}']")

    # 初始化并训练模型
    model = AdaBoostClassifier(n_estimators=n_estimators
                            , learning_rate=learning_rate
                            , algorithm=algorithm
                            , random_state=random_state
                            )
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# 逻辑回归
def train_logistic_regression(file_path=None
                            , data=None
                            , penalty='l2'
                            , C=1.0
                            , solver='lbfgs'
                            , max_iter=100
                            , test_size=0.2
                            , random_state=42
                            ,**kwargs):
    """
    训练逻辑回归分类器。

    参数说明：
    file_path : str, optional
        包含训练和测试数据的文件路径（如 .csv 文件）。若提供，则优先使用此路径加载数据。

    data : tuple (X_train, X_test, y_train, y_test), optional
        已经准备好的训练和测试数据。若提供，则跳过文件读取。

    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        正则化方式。常用选项包括：
            - 'l2'：岭回归（默认）
            - 'l1'：套索回归（需要合适的 solver）
            - 'elasticnet'：弹性网络（需配合 solver='saga'）
            - 'none'：无正则化

    C : float, default=1.0
        正则化强度的倒数，值越小表示正则化越强。通常在 [0.01, 100] 范围内调参。

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
        优化算法选择：
            - 'lbfgs'：适合中小型数据，支持 L2 正则化（默认）
            - 'liblinear'：支持 L1 和 L2，适合小数据集
            - 'saga'：支持所有 penalty，适合大规模稀疏数据
            - 'sag', 'newton-cg'：更适合 L2 正则，要求特征缩放

    max_iter : int, default=100
        最大迭代次数。若收敛失败，可以尝试调大。

    test_size : float, default=0.2
        测试集占总数据的比例，例如 0.2 表示 20%。

    random_state : int, default=42
        随机种子，用于确保数据划分的一致性和模型可复现性。
    """

    if data is not None:
        X_train, X_test, y_train, y_test = data
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path,
                                    data=data,test_size=test_size,random_state=random_state)

    print(f"使用逻辑回归分类器训练模型... [penalty='{penalty}', C={C}, solver='{solver}']")

    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter,
                            class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


# 训练 K 近邻（KNN）分类器
def train_kneighbors(file_path=None
                    , data=None
                    , n_neighbors=5
                    , weights='uniform'
                    , algorithm='auto'
                    , leaf_size=30
                    , p=2, metric='minkowski'
                    , test_size=0.2, random_state=42
                    ,**kwargs):
    """
    训练 K 近邻（KNN）分类器。

    参数:
        file_path: str，可选，数据文件路径
        data: tuple，可选，(X_train, X_test, y_train, y_test)
        n_neighbors: int，邻居数量
        weights: str，'uniform' 或 'distance'
        algorithm: str，'auto', 'ball_tree', 'kd_tree', 'brute'
        leaf_size: int，叶子节点大小（影响 BallTree 或 KDTree 性能）
        p: int，距离度量的幂参数（p=1 曼哈顿距离，p=2 欧几里得距离）
        metric: str，距离度量方式（默认 'minkowski'）
        test_size: float，测试集比例
        random_state: int，随机种子（仅用于数据划分）
    """
    # 获取数据
    if data is not None:
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path, 
                                                                    data=data,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    # 参数校验
    if weights not in ['uniform', 'distance']:
        raise ValueError("不支持的 weights，请选择 'uniform' 或 'distance'")
    if algorithm not in ['auto', 'ball_tree', 'kd_tree', 'brute']:
        raise ValueError("不支持的 algorithm，请选择 'auto', 'ball_tree', 'kd_tree' 或 'brute'")

    print(f"使用 K 近邻分类器训练模型... [n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}', p={p}, metric='{metric}']")

    # 初始化并训练模型
    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                weights=weights,
                                algorithm=algorithm,
                                leaf_size=leaf_size,
                                p=p,
                                metric=metric)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

# 训练随机森林分类器
def train_random_forest(file_path=None
                        , data=None
                        , n_estimators=100
                        , criterion='gini'
                        , max_depth=None
                        , min_samples_split=2
                        , min_samples_leaf=1
                        , max_features='sqrt'
                        , bootstrap=True
                        , test_size=0.2
                        , random_state=42
                        ,**kwargs):
    """
    训练随机森林分类器。

    参数:
        file_path: str，可选，数据文件路径
        data: tuple，可选，(X_train, X_test, y_train, y_test)
        n_estimators: int，树的数量
        criterion: str，划分标准，'gini' 或 'entropy'
        max_depth: int，树最大深度
        min_samples_split: int，内部节点划分最小样本数
        min_samples_leaf: int，叶子节点最小样本数
        max_features: str 或 int，分裂时考虑的最大特征数（'auto', 'sqrt', 'log2'等）
        bootstrap: bool，是否采用有放回抽样
        test_size: float，测试集比例
        random_state: int，随机种子
    """
    # 获取数据
    if data is not None:
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path, 
                                                                    data=data,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    # 参数校验
    if criterion not in ['gini', 'entropy', 'log_loss']:
        raise ValueError("不支持的 criterion，请选择 'gini'、'entropy' 或 'log_loss'")

    print(f"使用随机森林分类器训练模型... [n_estimators={n_estimators}, criterion='{criterion}', max_depth={max_depth}, bootstrap={bootstrap}]")

    # 初始化并训练模型
    model = RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                bootstrap=bootstrap,
                                random_state=random_state)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

# XGBoost分类器
def train_xgboost(file_path=None, data=None,
                n_estimators=100, learning_rate=0.1, max_depth=3, 
                subsample=1.0, colsample_bytree=1.0,
                test_size=0.2, random_state=42
                ,**kwargs):
    """
    训练 XGBoost 分类器。
    """
    if data is not None:
        X_train, X_test, y_train, y_test = data
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path,
                                                                    data=data,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    print(f"使用 XGBoost 分类器训练模型... [n_estimators={n_estimators}, max_depth={max_depth}]")

    model = XGBClassifier(n_estimators=n_estimators,
                          learning_rate=learning_rate,
                          max_depth=max_depth,
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          use_label_encoder=False,
                          eval_metric='logloss',
                          random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# CATBoost分类器
def train_catboost(file_path=None, data=None,
                   iterations=100, learning_rate=0.1, depth=6,
                   test_size=0.2, random_state=42
                   ,**kwargs):
    if data is not None:
        X_train, X_test, y_train, y_test = data
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(
            file_path=file_path, data=data, test_size=test_size, random_state=random_state)

    print(f"使用 CatBoost 分类器训练模型... [iterations={iterations}, depth={depth}]")
    model = CatBoostClassifier(iterations=iterations,
                               learning_rate=learning_rate,
                               depth=depth,
                               verbose=0,
                               random_seed=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# 训练 LDA 分类器。
def train_lda(file_path=None, data=None,
              solver='svd', shrinkage=None,
              test_size=0.2, random_state=42
              ,**kwargs):
    """
    训练 LDA 分类器。
    """
    if data is not None:
        X_train, X_test, y_train, y_test = data
    else:
        X_train, X_test, y_train, y_test = gnns.data.get_split_data(file_path=file_path,
                                                                    data=data,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    print(f"使用 LDA 分类器训练模型... [solver='{solver}', shrinkage={shrinkage}]")

    model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

