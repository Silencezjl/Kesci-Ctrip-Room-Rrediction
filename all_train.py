import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
import datetime
import os
from sklearn.model_selection import KFold
import lightgbm as lgb


def make_dir():
    if not os.path.exists('out'):
        os.makedirs('out')
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('model/stacking'):
        os.makedirs('model/stacking')
    if not os.path.exists('model/stacking/xgb'):
        os.makedirs('model/stacking/xgb')
    if not os.path.exists('data'):
        print('数据目录不存在，请检查是否存在data文件夹')
    if not os.path.exists('data/stack'):
        os.makedirs('data/stack')


def make_leak():
    '''
    穿越特征，用明天的来预测今天的
    :return: none
    '''
    sample_test = pd.read_csv("data/test_sample_small.txt", usecols=["basicroomid",
                                                                    "orderdate",
                                                                    "basic_30days_ordnumratio",
                                                                    "room_30days_ordnumratio", "roomid"], sep='\t')
    sample_train = pd.read_csv("data/train_sample_small.txt", usecols=["basicroomid",
                                                                      "orderdate",
                                                                      "basic_30days_ordnumratio",
                                                                      "room_30days_ordnumratio", "roomid"], sep='\t')
    test = pd.concat([sample_test, sample_train], ignore_index=True)

    basic_30days = test[["basicroomid", "orderdate", "basic_30days_ordnumratio"]]
    ss = basic_30days.drop_duplicates().sort_values("basicroomid").dropna()
    result = ss.set_index(["basicroomid", "orderdate"]).unstack()
    result.columns = result.columns.levels[1] + "basic30"
    result = result.reset_index()
    result.to_csv("data/basicroom.csv", index=None)

    room_30days = test[["roomid", "basicroomid", "orderdate", "room_30days_ordnumratio"]]
    ss = room_30days.drop_duplicates().sort_values("roomid").dropna()
    ss.roomid = ss.roomid + "&" + ss.basicroomid
    ss = ss.drop("basicroomid", axis=1)
    result2 = ss.set_index(["roomid", "orderdate"]).unstack()
    result2.columns = result2.columns.levels[1] + "room30"
    result2 = result2.reset_index()
    result2["basicroomid"] = result2.roomid.apply(lambda x: x.split("&")[1])
    result2["roomid"] = result2.roomid.apply(lambda x: x.split("&")[0])
    result2.to_csv("data/room.csv", index=None)


def make_focus(df):
    '''
    人工造特征
    以下特征主要通过三种方式衍生：
    1. 标明某特征是否是该orderid中最大(小)值
    2. 标明该房型特定服务是否满足该用户的需求
    3. 历史购买记录和本次差别的绝对值
    '''

    def lowest(ss):
        res = ss.copy()
        res[:] = 0
        res[ss == ss.min()] = 1
        return res

    def highest(ss):
        res = ss.copy()
        res[:] = 0
        res[ss == ss.max()] = 1
        return res

    def nearest(ss):
        tmp = ss.abs()
        res = ss.copy()
        res[:] = 0
        res[tmp == tmp.min()] = 1
        return res

    focus = df.loc[:, ['orderid', 'orderlabel', "basicroomid", "orderdate",
                       'roomid', 'roomtag_1', 'basic_comment_ratio', 'orderbehavior_9', 'star']].copy()
    ln_ordernum = np.log(df.user_ordernum + 1)
    focus['zzlowest_rank'] = df.groupby('orderid')['rank'].transform(highest)
    focus['zzlowest_price'] = df.groupby('orderid')['price_deduct'].transform(lowest)
    focus['zzhighest_start'] = df.groupby('orderid')['star'].transform(lowest)

    focus['zzhighest_basic30ratio'] = df.groupby('orderid')['basic_30days_ordnumratio'].transform(highest)
    focus['zzhighest_basic7ratio'] = df.groupby('orderid')['basic_week_ordernum_ratio'].transform(highest)

    focus['zznearest_rank'] = (df['rank'] - df['user_rank_ratio']).groupby(df['orderid']).transform(nearest)
    focus['zznearest_price'] = (df['price_deduct'] - df.user_avgprice + 140).groupby(df['orderid']).transform(nearest)
    focus['zznearest_start'] = (df.star - df.star_lastord).groupby(df['orderid']).transform(lowest)
    focus['zznearest_value'] = (df.returnvalue - df.return_lastord).groupby(df['orderid']).transform(highest)

    focus['nearest_rank'] = df['rank'] - df['user_rank_ratio']
    focus['nearest_price'] = df['price_deduct'] - df.user_avgprice
    focus['nearest_price_work'] = df['price_deduct'] - df.user_avgdealpriceworkday
    focus['nearest_price_max'] = df['price_deduct'] - df.user_maxprice

    focus['roomsv1'] = df.roomservice_1
    focus['roomsv2'] = (df.roomservice_2 == 1) & (df.user_roomservice_2_1ratio > 0.5)
    focus['roomsv3'] = (df.roomservice_3 > 0) & (df.user_roomservice_3_123ratio > 0.8)

    focus['roomsv4_2'] = (df.roomservice_4 == 2) & (df.user_roomservice_4_2ratio > 0.6)
    focus['roomsv4_0'] = (df.roomservice_4 == 0) & (df.user_roomservice_4_0ratio > 0.6)
    focus['roomsv4_3'] = (df.roomservice_4 == 3) & (df.user_roomservice_4_3ratio > 0.6)
    focus['roomsv4_4'] = (df.roomservice_4 == 4) & (df.user_roomservice_4_4ratio > 0.6)
    focus['roomsv4_5'] = (df.roomservice_4 == 5) & (df.user_roomservice_4_5ratio > 0.6)
    focus['roomsv4_1'] = (df.roomservice_4 == 1) & (df.user_roomservice_4_1ratio > 0.6)

    focus['roomsv5'] = (df.roomservice_5 == 1) & (df.user_roomservice_5_1ratio > 0.5)

    user_roomservice6_max_ratio = np.argmax([df.user_roomservice_6_0ratio, df.user_roomservice_6_1ratio,
                                             df.user_roomservice_6_2ratio])

    focus['roomsv6'] = df.roomservice_6 == user_roomservice6_max_ratio

    focus['roomsv7_0'] = (df.roomservice_7 == 0) & (df.user_roomservice_7_0ratio >= 0.5)

    focus['roomsv8_1'] = (df.roomservice_8 == 1) & (df.user_roomservice_8_1ratio >= 0.2)
    focus['roomsv8_345'] = (df.roomservice_8 > 2) & (df.user_roomservice_5_345ratio > 0.2)

    focus['roomservice2e'] = df.roomservice_2 == df.roomservice_2_lastord
    focus['roomservice3e'] = df.roomservice_3 == df.roomservice_3_lastord
    focus['roomservice4e'] = df.roomservice_4 == df.roomservice_4_lastord
    focus['roomservice5e'] = df.roomservice_5 == df.roomservice_5_lastord
    focus['roomservice6e'] = df.roomservice_6 == df.roomservice_6_lastord
    focus['roomservice8e'] = df.roomservice_8 == df.roomservice_8_lastord

    focus['roomtag2e'] = df.roomtag_2 == df.roomtag_2_lastord
    focus['roomtag3e'] = df.roomtag_3 == df.roomtag_3_lastord
    focus['roomtag4e'] = df.roomtag_4 == df.roomtag_4_lastord
    focus['roomtag5e'] = df.roomtag_5 == df.roomtag_5_lastord
    focus['roomtag6e'] = df.roomtag_6 == df.roomtag_6_lastord

    focus['hotelid_lastord_e'] = (df.hotelid_lastord == df.hotelid) * ln_ordernum

    focus['basicroomid_lastord_e'] = (df.basicroomid_lastord == df.basicroomid) * ln_ordernum
    focus['roomid_lastord_e'] = (df.roomid_lastord == df.roomid) * ln_ordernum
    focus['ranke'] = np.fabs(df['rank'] - df['rank_lastord'])
    focus['starte'] = np.fabs(df['star'] - df['star_lastord'])

    focus['diff_return'] = df.returnvalue - df.return_lastord
    focus['diff_star'] = df.star - df.star_lastord
    focus['diff_minprice'] = df.user_minprice - df.hotel_minprice_lastord
    focus['user_ordernum'] = df.user_ordernum  # 用户的订单总数
    focus['ordertype'] = np.argmax(
        [df.ordertype_1_ratio, df.ordertype_2_ratio, df.ordertype_3_ratio, df.ordertype_4_ratio,
         df.ordertype_5_ratio, df.ordertype_6_ratio, df.ordertype_7_ratio, df.ordertype_8_ratio,
         df.ordertype_9_ratio, df.ordertype_10_ratio, df.ordertype_11_ratio], axis=0)
    focus['room_30days_ordenumratio'] = df.room_30days_ordnumratio
    focus['basic_30days_ordenumratio'] = df.basic_30days_ordnumratio
    focus['basic_week_ordernum_ratio'] = df.basic_week_ordernum_ratio

    focus['basic_recent3_ordernum_ratio'] = df.basic_recent3_ordernum_ratio
    focus['room_30days_realratio'] = df.room_30days_realratio
    focus['orderbehavior_8'] = df.orderbehavior_8
    focus['basic_30days_realratio'] = df.basic_30days_realratio
    focus['user_confirmtime'] = df.user_confirmtime
    focus['user_avggoldstar'] = df.user_avggoldstar
    focus['user_avgadvanceddate'] = df.user_avgadvanceddate
    focus['ordertype_10_ratio'] = df.ordertype_10_ratio
    focus['basic_maxarea'] = df.basic_maxarea

    focus['user_avgrecommendlevel'] = df.user_avgrecommendlevel
    focus['user_minprice'] = df.user_minprice
    focus['orderbehavior_6_ratio'] = df.orderbehavior_6_ratio
    focus['user_avgstar'] = df.user_avgstar
    focus['orderbehavior_7_ratio'] = df.orderbehavior_7_ratio
    focus['user_avgprice_star'] = df.user_avgprice_star
    focus['user_stdprice'] = df.user_stdprice
    focus['user_cvprice'] = df.user_cvprice
    focus['user_maxprice'] = df.user_maxprice
    focus['user_roomservice_8_1ratio'] = df.user_roomservice_8_1ratio
    focus['orderbehavior_2_ratio'] = df.orderbehavior_2_ratio
    focus['user_roomservice_4_3ratio'] = df.user_roomservice_4_3ratio
    focus['user_roomservice_7_0ratio'] = df.user_roomservice_7_0ratio
    focus['user_roomservice_3_123ratio'] = df.user_roomservice_3_123ratio
    focus['user_avgprice'] = df.user_avgprice
    focus['ordertype_6_ratio'] = df.ordertype_6_ratio
    focus['roomtag_3'] = df.roomtag_3
    focus['user_roomservice_5_1ratio'] = df.user_roomservice_5_1ratio
    focus['price_last_lastord'] = df.price_last_lastord
    focus['user_avgdealpriceworkday'] = df.user_avgdealpriceworkday

    focus['user_avgdealpriceholiday'] = df.user_avgdealpriceholiday
    focus['orderbehavior_9'] = df.orderbehavior_9
    focus['user_avgroomnum'] = df.user_avgroomnum
    focus['ordertype_8_ratio'] = df.ordertype_8_ratio
    focus['user_roomservice_2_1ratio'] = df.user_roomservice_2_1ratio
    focus['user_avgpromotion'] = df.user_avgpromotion
    focus['roomservice_4'] = df.roomservice_4
    focus['user_medprice_3month'] = df.user_medprice_3month
    focus['user_rank_ratio'] = df.user_rank_ratio
    focus['returnvalue'] = df.returnvalue

    return focus


def use_leak(test):
    '''
    利用穿越特征
    :param test:
    :return:
    '''
    basicroomleak = pd.read_csv("data/basicroom.csv")
    roomleak = pd.read_csv("data/room.csv")

    data_small = test[["orderid", "roomid", "basicroomid", "orderdate"]]

    result_final = 0
    for i in data_small["orderdate"].drop_duplicates():
        if i == "2013-04-25":
            seek = data_small[data_small.orderdate == i]
            i = "2013-04-24"
            toda = str(pd.to_datetime(i)).split(" ")[0]
            tomo = str(pd.to_datetime(i) + pd.Timedelta("1 days")).split(" ")[0]
            seek = pd.merge(seek, basicroomleak[["basicroomid", toda + "basic30", tomo + "basic30"]],
                            on="basicroomid", how="left")  # 合并basicroomleak
            seek = pd.merge(seek, roomleak[["basicroomid", "roomid", toda + "room30", tomo + "room30"]],
                            on=["basicroomid", "roomid"], how="left")  # 合并roomleak
            seek["delta_basic"] = seek[tomo + "basic30"] - seek[toda + "basic30"]
            seek["delta_room"] = seek[tomo + "room30"] - seek[toda + "room30"]
            result = seek[["orderid", "roomid", "delta_basic", "delta_room"]]

        else:
            seek = data_small[data_small.orderdate == i]
            toda = str(pd.to_datetime(i)).split(" ")[0]
            tomo = str(pd.to_datetime(i) + pd.Timedelta("1 days")).split(" ")[0]
            seek = pd.merge(seek, basicroomleak[["basicroomid", toda + "basic30", tomo + "basic30"]],
                            on="basicroomid", how="left")  # 合并basicroomleak
            seek = pd.merge(seek, roomleak[["basicroomid", "roomid", toda + "room30", tomo + "room30"]],
                            on=["basicroomid", "roomid"], how="left")  # 合并roomleak
            seek["delta_basic"] = seek[tomo + "basic30"] - seek[toda + "basic30"]
            seek["delta_room"] = seek[tomo + "room30"] - seek[toda + "room30"]
            result = seek[["orderid", "roomid", "delta_basic", "delta_room"]]
        try:
            result_final = pd.concat([result_final, result], ignore_index=True)
        except:
            result_final = result

    return result_final


def get_oof_xgb(x_train, y_train, kf, ntrain):
    '''
    训练的主程序
    :param x_train:
    :param y_train:
    :param kf:
    :param ntrain:
    :return:
    '''
    if os.path.exists('data/stack/lgb_class_train2.h5'):
        store = pd.HDFStore('data/stack/lgb_class_train2.h5')
        oof_train = store["train"]
        store.close()
    else:
        oof_train = np.zeros((ntrain,))

        params = {
            "application": "binary",
            "learning_rate": 0.012,
            "max_depth": 8,
            "num_leaves": 128
        }

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            print(i)
            kf_X_train = x_train.values[train_index]
            kf_y_train = y_train.values[train_index]
            kf_X_test = x_train.values[test_index]

            if os.path.exists('model/stacking/xgb/lgb2_class'+str(i)+'.model'):
                clf = xgb.Booster(model_file='model/stacking/xgb/lgb2_class'+str(i)+'.model')
            else:
                clf = lgb.train(params, lgb.Dataset(kf_X_train, label=kf_y_train), 3000)
                clf.save_model('model/stacking/xgb/lgb2_class'+str(i)+'.model')

            oof_train[test_index] = clf.predict(kf_X_test)

        oof_train = oof_train.reshape(-1, 1)
        store= pd.HDFStore('data/stack/lgb_class_train2.h5')
        store["train2"] = pd.DataFrame(oof_train)
        store.close()

    return oof_train


if __name__ == '__main__':
    make_dir()

    make_leak()

    start_time = time.time()

    train = pd.read_csv('data/competition_train.txt', sep='\t')
    print(train.shape)
    train.star_lastord = train.star_lastord.fillna(train.user_avgstar)
    train.basic_minprice_lastord = train.basic_minprice_lastord.fillna(train.user_minprice)
    train.rank_lastord = train.rank_lastord.fillna(train.user_rank_ratio)
    train.return_lastord = train.return_lastord.fillna(train.user_avgpromotion)

    print("数据加载完成！cost time:", time.time()-start_time)
    train = make_focus(train)

    train = pd.merge(use_leak(train), train, on=['orderid', 'roomid'])

    print("特征重构完成！cost time:", time.time()-start_time)

    hold_orderid = train['orderid']
    hold_roomid = train['roomid']
    del train['orderid']
    del train['roomid']
    del train['basicroomid']
    del train['orderdate']
    train = train.astype('float64')
    X = train.drop(['orderlabel'], axis=1)
    y = train.orderlabel

    save_columns = [str(i) for i in X.columns]
    with open('model/save_columns.txt', 'w') as ff:
        ff.write(','.join(save_columns))

    # BGDT 造特征
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 划分数据集

    params_features = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,
        'max_depth': 3,
        'lambda': 0.7,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 2,
        'scale_pos_weight': 1.5,
        'silent': 0,
        'eta': 0.018,
        'seed': 11,
        'eval_metric': 'auc'
    }

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_test, label=y_test)
    # d_test = xgb.DMatrix(X_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # 造特征训练
    model_bst = xgb.train(params_features, d_train, 80, watchlist, verbose_eval=20)
    model_bst.save_model('model/feature_single.model')

    train_new_feature = model_bst.predict(xgb.DMatrix(X), pred_leaf=True)
    # test_new_feature = model_bst.predict(d_test, pred_leaf=True)

    train_new_feature1 = pd.DataFrame(train_new_feature)  # 训练集造出的新特征

    # 以上是造特征，以下是正常训练

    train = pd.DataFrame(pd.concat([X, train_new_feature1], axis=1))

    ntrain = train.shape[0]
    kf = KFold(n_splits=5, random_state=2000)

    print("xgb训练")
    data2 = get_oof_xgb(train, y, kf, ntrain)
    print("xgb训练完成")
