# -*- coding:utf-8 -*-
# @author: zhangxiang

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import gc

if __name__=="__main__":
    train = pd.read_csv('../data/train_feature01.csv', header=0)
    test = pd.read_csv('../data/test_feature01.csv', header=0)
    baseline = pd.read_csv('../data/test/orderFuture_test.csv', header=0)

    # print('the shape of train data is {}'.format(train.shape))
    # print('the shape of test data is {}'.format(test.shape))

    target = train['target']

    # 正负样本比率为1:5左右，对正样本进行过采样处理，是正负样本数据平衡    // 并没有显著的提升 auc score
    # for i in range(5):
    #     train.append(train[train['target']==1])

    train = train.drop(['userid', 'target'], axis=1)
    test = test.drop('userid', axis=1)
    num_train = len(train)

    data = pd.concat([train, test])

    # 填充NaN
    data['mean_rating'] = data['mean_rating'].fillna(data['mean_rating'].mean())
    data = data.fillna('missing')
    # preprocessing
    col2dum = ['gender', 'province', 'age', 'lastType', 'lastType2', 'lastType3', 'firstType']
    data_dummies = pd.get_dummies(data[col2dum])
    data = data.drop(col2dum, axis=1)
    data = pd.concat([data, data_dummies], axis=1)
    sta = StandardScaler()
    col2sta = ['type1_time', 'type2_time', 'type3_time', 'type4_time', 'type5_time', 'type6_time', 'type7_time',
               'type8_time', 'type9_time', 'dist_to_3', 'dist_to_5', 'dist_to_8', 'dist_to_9',
               'timespan_mean', 'timespan_var', 'timespan_min', 'timespan_last', 'timespan_last2', 'timespan_last3',
               'timespan_last4', 'timespan_first', 'timespan_last3_mean', 'timespan_last3_var',
               'var_5_ts', 'var_6_ts', 'var_7_ts', 'var_8_ts', 'var_9_ts', 'mean_5_ts', 'mean_6_ts', 'mean_7_ts',
               'mean_8_ts', 'mean_9_ts', 'min_2_ts', 'min_3_ts', 'min_4_ts', 'min_5_ts', 'min_6_ts',
               'min_7_ts', 'min_8_ts', 'max_5_ts', 'max_6_ts', 'max_7_ts', 'max_8_ts', 'mean_plus_var_9', 'mean_rating']
    data[col2sta] = sta.fit_transform(data[col2sta])
    train = data[:num_train]
    train_X, valid_X, train_y, valid_y = train_test_split(train, target, test_size=0.3, random_state=2017)
    train_X = np.array(train_X)
    train_y = train_y.values
    valid_X = np.array(valid_X)
    valid_y = valid_y.values
    test_X = np.array(data[num_train:])

    feature_name = train.columns.tolist()
    # for xgboost
    dtrain = xgb.DMatrix(data=train_X, label=train_y, feature_names=feature_name)     # 使用原始的特征名
    dvalid = xgb.DMatrix(data=valid_X, label=valid_y, feature_names=feature_name)
    dtest = xgb.DMatrix(data=test_X, feature_names=feature_name)
    watchlist = [(dtrain, 'train'),(dvalid, 'valid')]

    # 这里的参数是没调参之前的参数，可自行使用 GridSearch 调参
    params = {
        'eta':0.02,
        'max_depth':5,
        'subsample':0.8,
        'lambda':10,
        'colsample_bytree':0.9,
        'objective':'binary:logistic',
        'eval_metric':'auc',
        'seed':2017,
        'silent':True
    }

    model = xgb.train(params, dtrain, num_boost_round=4000, evals=watchlist, verbose_eval=50, early_stopping_rounds=100)
    predict = model.predict(dtest)
    baseline['orderType'] = predict
    baseline.to_csv('../data/baseline_for_xgboost01.csv', index=False, encoding='utf-8')

    feature_importance = model.get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x:x[1], reverse=True)
    fs = []
    for (feature,importance) in feature_importance:
        fs.append('{0},{1}\n'.format(feature, importance))
    with open('../data/feature_importance.csv', 'w') as f:
        f.writelines('feature,score\n')
        f.writelines(fs)

