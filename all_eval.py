import numpy as np
import pandas as pd
import xgboost as xgb
import time
import os
import lightgbm as lgb


def use_leak(test):
    basicroomleak = pd.read_csv("data/basicroom.csv")
    roomleak = pd.read_csv("data/room.csv")

    data_small = test[["orderid", "roomid", "basicroomid", "orderdate"]]

    result_final = 0
    data_date = data_small["orderdate"].drop_duplicates()
    for i in data_date:
        # print(i)
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
            print(seek.shape)
            toda = str(pd.to_datetime(i)).split(" ")[0]
            tomo = str(pd.to_datetime(i) + pd.Timedelta("1 days")).split(" ")[0]
            seek = pd.merge(seek, basicroomleak[["basicroomid", toda + "basic30", tomo + "basic30"]],
                            on="basicroomid", how="left")  # 合并basicroomleak
            # print('1', seek.shape)
            seek = pd.merge(seek, roomleak[["basicroomid", "roomid", toda + "room30", tomo + "room30"]],
                            on=["basicroomid", "roomid"], how="left")  # 合并roomleak
            # print('2', seek.shape)
            seek["delta_basic"] = seek[tomo + "basic30"] - seek[toda + "basic30"]
            seek["delta_room"] = seek[tomo + "room30"] - seek[toda + "room30"]
            result = seek[["orderid", "roomid", "delta_basic", "delta_room"]]
            print('result_shape', result.shape)
        try:
            result_final = pd.concat([result_final, result], ignore_index=True)
        except:
            result_final = result
        # print(result.shape)


    return result_final


def make_focus(df):
    '''
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

    focus = df.loc[:, ['orderid', "basicroomid", "orderdate",
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


def first_lgb_class_pre(X_test):
    '''xgb pre'''
    ntest = X_test.shape[0]
    if os.path.exists('data/stack/lgb_class_pre.h5'):
        store = pd.HDFStore('data/stack/lgb_class_pre.h5')
        oof_test_lgb_class = store["pre"]
        store.close()

    else:
        # list_lgb_class=os.listdir('model/stacking/xgb')
        oof_test_lgb_class = np.zeros((ntest,))
        oof_test_skf_lgb_class = np.empty((5, ntest))

        for i in range(5):
            model = lgb.Booster(model_file=str('model/stacking/xgb/lgb2_class' + str(i) + '.model'))
            oof_test_skf_lgb_class[i, :] = model.predict(X_test)
        oof_test_lgb_class[:] = oof_test_skf_lgb_class.mean(axis=0)
        oof_test_lgb_class = oof_test_lgb_class.reshape(-1, 1)
        # save pre
        store = pd.HDFStore('data/stack/lgb_class_pre.h5')
        store["pre"] = pd.DataFrame(oof_test_lgb_class)
        store.close()
    return oof_test_lgb_class


if __name__ == '__main__':
    start_time = time.time()
    tests = pd.read_csv('data/competition_test.txt', sep='\t')

    tests.star_lastord = tests.star_lastord.fillna(tests.user_avgstar)
    tests.basic_minprice_lastord = tests.basic_minprice_lastord.fillna(tests.user_minprice)
    tests.rank_lastord = tests.rank_lastord.fillna(tests.user_rank_ratio)
    tests.return_lastord = tests.return_lastord.fillna(tests.user_avgpromotion)
    print("数据加载完成！cost time:", time.time()-start_time)

    tests = make_focus(tests)
    tests = pd.merge(use_leak(tests), tests, on=['orderid', 'roomid'])
    print("特征重构完成！cost time:", time.time()-start_time)

    hold_roomid = tests['roomid']
    hold_orderid = tests['orderid']
    del tests['orderid']
    del tests['roomid']
    del tests['basicroomid']
    del tests['orderdate']
    tests = tests.astype('float64')
    with open('model/save_columns.txt', 'r') as ff:
        save_columns = ff.readline().replace('\n', '').split(',')
    assert len(tests.columns) == len(save_columns)

    feature_bst = xgb.Booster()
    feature_bst.load_model("model/feature_single.model")

    test_new_feature = feature_bst.predict(xgb.DMatrix(tests), pred_leaf=True)
    #
    test_new_feature1 = pd.DataFrame(test_new_feature)
    tests = pd.DataFrame(pd.concat([tests, test_new_feature1], axis=1))
    print(tests.corr())
    store2 = pd.HDFStore('../Ctrip/data/tests_afterGBDT.h5')
    tests2 = store2['test_afterGBDT']
    store2.close()
    del tests2['orderid']
    del tests2['roomid']
    print(tests2.corr())
    tests.corr().to_csv('test.csv')
    tests2.corr().to_csv('test2.csv')
    exit()
    data2 = first_lgb_class_pre(tests)
    print('lgb_class预测完成')

    final_pre_all = pd.concat([hold_orderid, hold_roomid, pd.DataFrame(data2)], axis=1)
    final_pre_all.columns = ['orderid', 'predict_roomid', 'preds']

    result = final_pre_all
    result.to_csv('out/final_res.csv', index=False)

    result = result.sort_values(["preds"], ascending=False)

    result = result.drop_duplicates(['orderid'])

    del result['preds']

    result.to_csv('out/final_commit.csv', index=False, header=['orderid', 'predict_roomid'])
