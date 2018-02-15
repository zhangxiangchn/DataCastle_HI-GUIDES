# -*- coding:utf-8 -*-
# @author: zhangxiang
import pandas as pd
import numpy as np

def get_type(df, userid):
    """
    获得特定次数行为
    :param df: train_A or test_A
    :param userid: userid to use
    :return: 特定次数的行为结果
    """
    lastType = 0
    lastType2 = 0
    lastType3 = 0
    firstType = 0
    action_num = len(df[df['userid']==userid])
    if action_num>=1:
        lastType = df[df['userid']==userid].iloc[-1]['actionType']
    if action_num>=2:
        lastType2 = df[df['userid']==userid].iloc[-2]['actionType']
    if action_num>=3:
        lastType3 = df[df['userid']==userid].iloc[-3]['actionType']
    if action_num>=4:
        firstType = df[df['userid']==userid].iloc[0]['actionType']
    return lastType, lastType2, lastType3, firstType

def get_time(df, userid):
    """
    获得距离各个行为最近的时间
    :param df: train_A, test_A
    :param userid: userid to use
    :return: 距离各个行为最近的时间
    """
    type1_time = 0
    type2_time = 0
    type3_time = 0
    type4_time = 0
    type5_time = 0
    type6_time = 0
    type7_time = 0
    type8_time = 0
    type9_time = 0
    userid_df = df[df['userid'] == userid]
    if len(userid_df)>=1:
        if len(userid_df[userid_df['actionType']==1])>0:
            type1_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==1].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==2])>0:
            type2_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==2].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==3])>0:
            type3_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==3].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==4])>0:
            type4_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==4].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==5])>0:
            type5_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==5].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==6])>0:
            type6_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==6].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==7])>0:
            type7_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==7].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==8])>0:
            type8_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==8].iloc[-1]['actionTime']
        if len(userid_df[userid_df['actionType']==9])>0:
            type9_time = userid_df.iloc[-1]['actionTime'] - userid_df[userid_df['actionType']==9].iloc[-1]['actionTime']
    return type1_time, type2_time, type3_time, type4_time, type5_time, type6_time, type7_time, type8_time, type9_time


def get_distance(df, userid):
    """ 求距离最近的9,3,5,8的距离"""
    distance_to_3 = 0
    distance_to_5 = 0
    distance_to_8 = 0
    distance_to_9 = 0
    dist_df = df[df['userid'] == userid]
    len_act = len(dist_df)
    listType = dist_df['actionType'].tolist()
    if len_act >= 1:
        if len(dist_df[dist_df['actionType'] == 3]) > 0:
            findloc = [a for a in range(len_act) if listType[a] == 3][-1]
            distance_to_3 = int(len_act - findloc)
        if len(dist_df[dist_df['actionType'] == 5]) > 0:
            findloc = [a for a in range(len_act) if listType[a] == 5][-1]
            distance_to_5 = int(len_act - findloc)
        if len(dist_df[dist_df['actionType'] == 8]) > 0:
            findloc = [a for a in range(len_act) if listType[a] == 8][-1]
            distance_to_8 = int(len_act - findloc)
        if len(dist_df[dist_df['actionType'] == 9]) > 0:
            findloc = [a for a in range(len_act) if listType[a] == 9][-1]
            distance_to_9 = int(len_act - findloc)

    return distance_to_3, distance_to_5, distance_to_8, distance_to_9

def get_time_span(df, userid):
    """ 计算对应的userid的时间间隔相关的数据 """
    timespan_mean = 0
    timespan_var = 0
    timespan_min = 0
    timespan_last = 0
    timespan_last2 = 0
    timespan_last3 = 0
    timespan_last4 = 0
    timespan_first = 0
    timespan_last3_mean = 0
    timespan_last3_var = 0
    var_5_ts = 0
    var_6_ts = 0
    var_7_ts = 0
    var_8_ts = 0
    var_9_ts = 0
    mean_5_ts = 0
    mean_6_ts = 0
    mean_7_ts = 0
    mean_8_ts = 0
    mean_9_ts = 0
    min_2_ts = 0
    min_3_ts = 0
    min_4_ts = 0
    min_5_ts = 0
    min_6_ts = 0
    min_7_ts = 0
    min_8_ts = 0
    max_5_ts = 0
    max_6_ts = 0
    max_7_ts = 0
    max_8_ts = 0
    mean_plus_var_9 = 0
    userdf = df[df['userid']==userid]
    listType = userdf['actionType'].tolist()
    if len(userdf)>=2:    # 前提
        ts = userdf['actionTime'].diff().tolist()[1:]
        if len(ts)>=1:
            timespan_mean = np.mean(ts)
            timespan_var = np.mean(ts)
            timespan_min = np.min(ts)
            timespan_last = ts[-1]
            timespan_first = ts[0]
        if len(ts)>=2:
            timespan_last2 = ts[-2]
        if len(ts)>=3:
            timespan_last3 = ts[-3]
            timespan_last3_mean = np.mean(ts[-3:])
            timespan_last3_var = np.var(ts[-3:])
        if len(ts)>=4:
            timespan_last4 = ts[-4]
        if len(userdf[userdf['actionType']==5])>=1:
            location5 = [a for a in range(len(userdf)) if listType[a]==5][-1]  # 找到最近的5的位置
            if (len(userdf)-location5)>1:   # 保证5不是最后一个Type
                var_5_ts = np.var(ts[location5:])
                mean_5_ts = np.mean(ts[location5:])
                min_5_ts = np.min(ts[location5:])
                max_5_ts = np.max(ts[location5:])
        if len(userdf[userdf['actionType'] == 6]) >= 1:
            location6 = [a for a in range(len(userdf)) if listType[a] == 6][-1]
            if (len(userdf) - location6) > 1:
                var_6_ts = np.var(ts[location6:])
                mean_6_ts = np.mean(ts[location6:])
                min_6_ts = np.min(ts[location6:])
                max_6_ts = np.max(ts[location6:])
        if len(userdf[userdf['actionType'] == 7]) >= 1:
            location7 = [a for a in range(len(userdf)) if listType[a] == 7][-1]
            if (len(userdf) - location7) > 1:
                var_7_ts = np.var(ts[location7:])
                mean_7_ts = np.mean(ts[location7:])
                min_7_ts = np.min(ts[location7:])
                max_7_ts = np.max(ts[location7:])
        if len(userdf[userdf['actionType'] == 8]) >= 1:
            location8 = [a for a in range(len(userdf)) if listType[a] == 8][-1]
            if (len(userdf) - location8) > 1:
                var_8_ts = np.var(ts[location8:])
                mean_8_ts = np.mean(ts[location8:])
                min_8_ts = np.min(ts[location8:])
                max_8_ts = np.max(ts[location8:])
        if len(userdf[userdf['actionType'] == 9]) >= 1:
            location9 = [a for a in range(len(userdf)) if listType[a] == 9][-1]
            if (len(userdf) - location9) > 1:
                var_9_ts = np.var(ts[location9:])
                mean_9_ts = np.mean(ts[location9:])
                mean_plus_var_9 = var_9_ts * mean_9_ts
        if len(userdf[userdf['actionType'] == 2]) >= 1:
            location2 = [a for a in range(len(userdf)) if listType[a] == 2][-1]
            if (len(userdf) - location2) > 1:
                min_2_ts = np.min(ts[location2:])
        if len(userdf[userdf['actionType'] == 3]) >= 1:
            location3 = [a for a in range(len(userdf)) if listType[a] == 3][-1]
            if (len(userdf) - location3) > 1:
                min_3_ts = np.min(ts[location3:])
        if len(userdf[userdf['actionType'] == 4]) >= 1:
            location4 = [a for a in range(len(userdf)) if listType[a] == 4][-1]
            if (len(userdf) - location4) > 1:
                min_4_ts = np.min(ts[location4:])
    return timespan_mean, timespan_var, timespan_min, timespan_last, timespan_last2, timespan_last3, timespan_last4, timespan_first, timespan_last3_mean, timespan_last3_var, \
           var_5_ts, var_6_ts, var_7_ts, var_8_ts, var_9_ts, mean_5_ts, mean_6_ts, mean_7_ts, mean_8_ts, mean_9_ts, min_2_ts, min_3_ts, min_4_ts, min_5_ts, min_6_ts, min_7_ts, \
           min_8_ts, max_5_ts, max_6_ts, max_7_ts, max_8_ts, mean_plus_var_9

if __name__ == "__main__":
    # training data
    train_A = pd.read_csv('../data/train/action_train.csv', header=0)
    train_F = pd.read_csv('../data/train/orderFuture_train.csv', header=0)
    train_H = pd.read_csv('../data/train/orderHistory_train.csv', header=0)
    train_C = pd.read_csv('../data/train/userComment_train.csv', header=0)
    train_P = pd.read_csv('../data/train/userProfile_train.csv', header=0)
    # test data
    test_A = pd.read_csv('../data/test/action_test.csv', header=0)
    test_F = pd.read_csv('../data/test/orderFuture_test.csv', header=0)
    test_H = pd.read_csv('../data/test/orderHistory_test.csv', header=0)
    test_C = pd.read_csv('../data/test/userComment_test.csv', header=0)
    test_P = pd.read_csv('../data/test/userProfile_test.csv', header=0)

    train = train_F
    test = test_F

    train = pd.merge(train, train_P, on='userid', how='left')
    test = pd.merge(test, test_P, on='userid', how='left')
    train.rename(columns={'orderType':'target'}, inplace=True)

    a1 = set(train_H[train_H['orderType']==1]['userid'].tolist())
    a2 = set(test_H[test_H['orderType']==1]['userid'].tolist())
    train['ever_buy'] = train['userid'].apply(lambda x:1 if x in a1 else 0)
    test['ever_buy'] = test['userid'].apply(lambda x:1 if x in a2 else 0)

    # 每个用户总的点击次数
    train_click_count = train_A.groupby(by='userid')['actionType'].count().to_dict()
    test_click_count = test_A.groupby(by='userid')['actionType'].count().to_dict()
    # 每个用户点击各个Type的次数
    train_click_ = train_A.groupby(by='userid')['actionType'].value_counts().to_dict()
    test_click_ = test_A.groupby(by='userid')['actionType'].value_counts().to_dict()
    # 统计每一种点击次数的占比
    for i in range(1, 10):
        train['click_{}'.format(i)] = train['userid'].apply(lambda x: float(train_click_[(x, i)]/train_click_count[x]) if (x,i) in train_click_.keys() else 0)
        test['click_{}'.format(i)] = test['userid'].apply(lambda x: float(test_click_[(x,i)]/test_click_count[x]) if (x,i) in test_click_.keys() else 0)

    # userid 的数量
    usercount_train = len(train_P['userid'])
    usercount_test = len(test_P['userid'])

    # 获得特定次数行为 get_type
    # train
    lastType = np.zeros(usercount_train)
    lastType2 = np.zeros(usercount_train)
    lastType3 = np.zeros(usercount_train)
    firstType = np.zeros(usercount_train)
    for i in range(usercount_train):
        lastType[i], lastType2[i], lastType3[i], firstType[i] = get_type(train_A, train_P['userid'][i])
    train['lastType'] = lastType
    train['lastType2'] = lastType2
    train['lastType3'] = lastType3
    train['firstType'] = firstType

    # test
    lastType = np.zeros(usercount_test)
    lastType2 = np.zeros(usercount_test)
    lastType3 = np.zeros(usercount_test)
    firstType = np.zeros(usercount_test)
    for i in range(usercount_test):
        lastType[i], lastType2[i], lastType3[i], firstType[i] = get_type(test_A, test_P['userid'][i])
    test['lastType'] = lastType
    test['lastType2'] = lastType2
    test['lastType3'] = lastType3
    test['firstType'] = firstType

    # 获得距离各个行为最近的时间 get_time
    # train
    type1_time = np.zeros(usercount_train)
    type2_time = np.zeros(usercount_train)
    type3_time = np.zeros(usercount_train)
    type4_time = np.zeros(usercount_train)
    type5_time = np.zeros(usercount_train)
    type6_time = np.zeros(usercount_train)
    type7_time = np.zeros(usercount_train)
    type8_time = np.zeros(usercount_train)
    type9_time = np.zeros(usercount_train)
    for i in range(usercount_train):
        type1_time[i], type2_time[i], type3_time[i], type4_time[i], type5_time[i], type6_time[i], type7_time[i], \
        type8_time[i], type9_time[i] = get_time(train_A, train_P['userid'][i])
    train['type1_time'] = type1_time
    train['type2_time'] = type2_time
    train['type3_time'] = type3_time
    train['type4_time'] = type4_time
    train['type5_time'] = type5_time
    train['type6_time'] = type6_time
    train['type7_time'] = type7_time
    train['type8_time'] = type8_time
    train['type9_time'] = type9_time

    # test
    type1_time = np.zeros(usercount_test)
    type2_time = np.zeros(usercount_test)
    type3_time = np.zeros(usercount_test)
    type4_time = np.zeros(usercount_test)
    type5_time = np.zeros(usercount_test)
    type6_time = np.zeros(usercount_test)
    type7_time = np.zeros(usercount_test)
    type8_time = np.zeros(usercount_test)
    type9_time = np.zeros(usercount_test)
    for i in range(usercount_test):
        type1_time[i], type2_time[i], type3_time[i], type4_time[i], type5_time[i], type6_time[i], type7_time[i], \
        type8_time[i], type9_time[i] = get_time(test_A, test_P['userid'][i])
    test['type1_time'] = type1_time
    test['type2_time'] = type2_time
    test['type3_time'] = type3_time
    test['type4_time'] = type4_time
    test['type5_time'] = type5_time
    test['type6_time'] = type6_time
    test['type7_time'] = type7_time
    test['type8_time'] = type8_time
    test['type9_time'] = type9_time


    # 求距离最近的9,3,5,8的距离 get_distance
    # train
    distance_to_3 = np.zeros(usercount_train)
    distance_to_5 = np.zeros(usercount_train)
    distance_to_8 = np.zeros(usercount_train)
    distance_to_9 = np.zeros(usercount_train)
    for i in range(usercount_train):
        distance_to_3[i], distance_to_5[i], distance_to_8[i], distance_to_9[i] = get_distance(train_A,
                                                                                              train_P['userid'][i])
    train['dist_to_3'] = distance_to_3
    train['dist_to_5'] = distance_to_5
    train['dist_to_8'] = distance_to_8
    train['dist_to_9'] = distance_to_9

    # test
    distance_to_3 = np.zeros(usercount_test)
    distance_to_5 = np.zeros(usercount_test)
    distance_to_8 = np.zeros(usercount_test)
    distance_to_9 = np.zeros(usercount_test)
    for i in range(usercount_test):
        distance_to_3[i], distance_to_5[i], distance_to_8[i], distance_to_9[i] = get_distance(test_A,
                                                                                              test_P['userid'][i])
    test['dist_to_3'] = distance_to_3
    test['dist_to_5'] = distance_to_5
    test['dist_to_8'] = distance_to_8
    test['dist_to_9'] = distance_to_9

    # 计算对应的userid的时间间隔相关的数据 get_time_span
    # train
    timespan_mean = np.zeros(usercount_train)
    timespan_var = np.zeros(usercount_train)
    timespan_min = np.zeros(usercount_train)
    timespan_last = np.zeros(usercount_train)
    timespan_last2 = np.zeros(usercount_train)
    timespan_last3 = np.zeros(usercount_train)
    timespan_last4 = np.zeros(usercount_train)
    timespan_first = np.zeros(usercount_train)
    timespan_last3_mean = np.zeros(usercount_train)
    timespan_last3_var = np.zeros(usercount_train)
    var_5_ts = np.zeros(usercount_train)
    var_6_ts = np.zeros(usercount_train)
    var_7_ts = np.zeros(usercount_train)
    var_8_ts = np.zeros(usercount_train)
    var_9_ts = np.zeros(usercount_train)
    mean_5_ts = np.zeros(usercount_train)
    mean_6_ts = np.zeros(usercount_train)
    mean_7_ts = np.zeros(usercount_train)
    mean_8_ts = np.zeros(usercount_train)
    mean_9_ts = np.zeros(usercount_train)
    min_2_ts = np.zeros(usercount_train)
    min_3_ts = np.zeros(usercount_train)
    min_4_ts = np.zeros(usercount_train)
    min_5_ts = np.zeros(usercount_train)
    min_6_ts = np.zeros(usercount_train)
    min_7_ts = np.zeros(usercount_train)
    min_8_ts = np.zeros(usercount_train)
    max_5_ts = np.zeros(usercount_train)
    max_6_ts = np.zeros(usercount_train)
    max_7_ts = np.zeros(usercount_train)
    max_8_ts = np.zeros(usercount_train)
    mean_plus_var_9 = np.zeros(usercount_train)
    for i in range(usercount_train):
        timespan_mean[i], timespan_var[i], timespan_min[i], timespan_last[i], timespan_last2[i], timespan_last3[i], \
        timespan_last4[i], timespan_first[i], timespan_last3_mean[i], timespan_last3_var[i], var_5_ts[i], \
        var_6_ts[i], var_7_ts[i], var_8_ts[i], var_9_ts[i], mean_5_ts[i], mean_6_ts[i], mean_7_ts[i], mean_8_ts[i], \
        mean_9_ts[i], min_2_ts[i], min_3_ts[i], min_4_ts[i], min_5_ts[i], min_6_ts[i], min_7_ts[i], min_8_ts[i], \
        max_5_ts[i], max_6_ts[i], max_7_ts[i], max_8_ts[i], mean_plus_var_9[i] = get_time_span(train_A,
                                                                                               train_P['userid'][i])
    train['timespan_mean'] = timespan_mean
    train['timespan_var'] = timespan_var
    train['timespan_min'] = timespan_min
    train['timespan_last'] = timespan_last
    train['timespan_last2'] = timespan_last2
    train['timespan_last3'] = timespan_last3
    train['timespan_last4'] = timespan_last4
    train['timespan_first'] = timespan_first
    train['timespan_last3_mean'] = timespan_last3_mean
    train['timespan_last3_var'] = timespan_last3_var
    train['var_5_ts'] = var_5_ts
    train['var_6_ts'] = var_6_ts
    train['var_7_ts'] = var_7_ts
    train['var_8_ts'] = var_8_ts
    train['var_9_ts'] = var_9_ts
    train['mean_5_ts'] = mean_5_ts
    train['mean_6_ts'] = mean_6_ts
    train['mean_7_ts'] = mean_7_ts
    train['mean_8_ts'] = mean_8_ts
    train['mean_9_ts'] = mean_9_ts
    train['min_2_ts'] = min_2_ts
    train['min_3_ts'] = min_3_ts
    train['min_4_ts'] = min_4_ts
    train['min_5_ts'] = min_5_ts
    train['min_6_ts'] = min_6_ts
    train['min_7_ts'] = min_7_ts
    train['min_8_ts'] = min_8_ts
    train['max_5_ts'] = max_5_ts
    train['max_6_ts'] = max_6_ts
    train['max_7_ts'] = max_7_ts
    train['max_8_ts'] = max_8_ts
    train['mean_plus_var_9'] = mean_plus_var_9

    # test
    timespan_mean = np.zeros(usercount_test)
    timespan_var = np.zeros(usercount_test)
    timespan_min = np.zeros(usercount_test)
    timespan_last = np.zeros(usercount_test)
    timespan_last2 = np.zeros(usercount_test)
    timespan_last3 = np.zeros(usercount_test)
    timespan_last4 = np.zeros(usercount_test)
    timespan_first = np.zeros(usercount_test)
    timespan_last3_mean = np.zeros(usercount_test)
    timespan_last3_var = np.zeros(usercount_test)
    var_5_ts = np.zeros(usercount_test)
    var_6_ts = np.zeros(usercount_test)
    var_7_ts = np.zeros(usercount_test)
    var_8_ts = np.zeros(usercount_test)
    var_9_ts = np.zeros(usercount_test)
    mean_5_ts = np.zeros(usercount_test)
    mean_6_ts = np.zeros(usercount_test)
    mean_7_ts = np.zeros(usercount_test)
    mean_8_ts = np.zeros(usercount_test)
    mean_9_ts = np.zeros(usercount_test)
    min_2_ts = np.zeros(usercount_test)
    min_3_ts = np.zeros(usercount_test)
    min_4_ts = np.zeros(usercount_test)
    min_5_ts = np.zeros(usercount_test)
    min_6_ts = np.zeros(usercount_test)
    min_7_ts = np.zeros(usercount_test)
    min_8_ts = np.zeros(usercount_test)
    max_5_ts = np.zeros(usercount_test)
    max_6_ts = np.zeros(usercount_test)
    max_7_ts = np.zeros(usercount_test)
    max_8_ts = np.zeros(usercount_test)
    mean_plus_var_9 = np.zeros(usercount_test)
    for i in range(usercount_test):
        timespan_mean[i], timespan_var[i], timespan_min[i], timespan_last[i], timespan_last2[i], timespan_last3[i], \
        timespan_last4[i], timespan_first[i], timespan_last3_mean[i], timespan_last3_var[i], var_5_ts[i], \
        var_6_ts[i], var_7_ts[i], var_8_ts[i], var_9_ts[i], mean_5_ts[i], mean_6_ts[i], mean_7_ts[i], mean_8_ts[i], \
        mean_9_ts[i], min_2_ts[i], min_3_ts[i], min_4_ts[i], min_5_ts[i], min_6_ts[i], min_7_ts[i], min_8_ts[i], \
        max_5_ts[i], max_6_ts[i], max_7_ts[i], max_8_ts[i], mean_plus_var_9[i] = get_time_span(test_A,
                                                                                               test_P['userid'][i])
    test['timespan_mean'] = timespan_mean
    test['timespan_var'] = timespan_var
    test['timespan_min'] = timespan_min
    test['timespan_last'] = timespan_last
    test['timespan_last2'] = timespan_last2
    test['timespan_last3'] = timespan_last3
    test['timespan_last4'] = timespan_last4
    test['timespan_first'] = timespan_first
    test['timespan_last3_mean'] = timespan_last3_mean
    test['timespan_last3_var'] = timespan_last3_var
    test['var_5_ts'] = var_5_ts
    test['var_6_ts'] = var_6_ts
    test['var_7_ts'] = var_7_ts
    test['var_8_ts'] = var_8_ts
    test['var_9_ts'] = var_9_ts
    test['mean_5_ts'] = mean_5_ts
    test['mean_6_ts'] = mean_6_ts
    test['mean_7_ts'] = mean_7_ts
    test['mean_8_ts'] = mean_8_ts
    test['mean_9_ts'] = mean_9_ts
    test['min_2_ts'] = min_2_ts
    test['min_3_ts'] = min_3_ts
    test['min_4_ts'] = min_4_ts
    test['min_5_ts'] = min_5_ts
    test['min_6_ts'] = min_6_ts
    test['min_7_ts'] = min_7_ts
    test['min_8_ts'] = min_8_ts
    test['max_5_ts'] = max_5_ts
    test['max_6_ts'] = max_6_ts
    test['max_7_ts'] = max_7_ts
    test['max_8_ts'] = max_8_ts
    test['mean_plus_var_9'] = mean_plus_var_9

    # 平均评分   存在NAN
    mean_rat_train = dict(train_C.groupby('userid')['rating'].mean())
    mean_rat_test = dict(test_C.groupby('userid')['rating'].mean())
    train['mean_rating'] = train['userid'].apply(lambda x: mean_rat_train[x] if x in mean_rat_train.keys() else np.nan)
    test['mean_rating'] = test['userid'].apply(lambda x: mean_rat_test[x] if x in mean_rat_test.keys() else np.nan)

    train['good'] = train['userid'].apply(
        lambda x: 1 if (x in mean_rat_train.keys() and mean_rat_train[x] < 4.9) else 0)
    test['good'] = test['userid'].apply(lambda x: 1 if (x in mean_rat_test.keys() and mean_rat_test[x] < 4.9) else 0)

    train.to_csv('../data/train_feature01.csv', index=False, encoding='utf-8')
    test.to_csv('../data/test_feature01.csv', index=False, encoding='utf-8')