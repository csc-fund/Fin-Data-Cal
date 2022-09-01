import numpy as np
import pandas as pd
from tqdm import tqdm
import time

##################################################################
# 参数
##################################################################
LAG_NUM = 20  # 历史信息数量
LAG_PERIOD = 4  # 当前交易日滞后期
OBS_J = 3  # 观测期 : 用于历史信息的平均预测和线性预测
PRE_K = 1  # 预测期
assert OBS_J < LAG_PERIOD < LAG_NUM

##################################################################
# 1.转换分红的面板数据 按照股票名存储历史信息
##################################################################
DIV_TABLE = pd.read_parquet('AShareDividend.parquet',
                            columns=['s_div_progress', 'stockcode', 'report_period', 'ann_date',
                                     'cash_dvd_per_sh_pre_tax', 's_div_baseshare'])  # 读取原始数据 筛选计算列
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总分红
# 按照stockcode升序后,再按照ann_date降序 # 取每个股票最近N个历史数据
DIV_TABLE = (DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0]).groupby(['stockcode']).head(LAG_NUM))
DIV_TABLE['ANNDATE_MAX'] = DIV_TABLE.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
DIV_P_TABLE = pd.pivot_table(DIV_TABLE, index=['stockcode'], columns=['ANNDATE_MAX'],
                             values=['ann_date', 'report_period', 'dvd_pre_tax']).fillna(0)  # 转置:按照信息排序后转置
DIV_P_TABLE.columns = [i[0] + '_{}'.format(i[1]) for i in DIV_P_TABLE.columns]  # 重命名列名
#
# col_list = list(
#     map(lambda x, y: str(x) + str(y),
#         ['ann_date_'] * LAG_NUM + ['report_period_'] * LAG_NUM, [i for i in range(LAG_NUM)] * 2))
# dtype_col = dict(zip(col_list, ['uint32'] * len(col_list)))
# DIV_P_TABLE = DIV_P_TABLE.astype(dtype_col)
del DIV_TABLE

# df_0.sort_values(by='ann_date', ascending=False, inplace=True)
# df_t = df_0[['stockcode', 'ann_date']].astype({'stockcode': 'category'})  # 前两列标识列

##################################################################
# 2.合并MV_TABLE表与DIV_P_TABLE表 df_0
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet', columns=['stockcode', 'ann_date', ]).astype(
    {'ann_date': 'uint32', }).set_index('stockcode')

##################################################################
# 测试数据---600738.SH在2018年有3次分红 已核对
##################################################################
# MV_TABLE = MV_TABLE[MV_TABLE.index == '600738.SH'].sort_values(by='ann_date', ascending=False)

# merge不创建numpy快,join创建numpy快
st = time.time()
print('start', time.time() - st)
np_code = MV_TABLE.index.to_numpy().reshape(MV_TABLE.shape[0], 1)
np_trade_date = np.tile(MV_TABLE.iloc[:, 0].to_numpy('<u4'), (LAG_NUM, 1)).T  # trade_date
MV_TABLE = MV_TABLE.iloc[:, :0]  # 移除trade_date
np_ann_date = MV_TABLE.join(DIV_P_TABLE.iloc[:, LAG_NUM * 0:LAG_NUM * 1], how='left').to_numpy('<u4')  # ann_date
np_dvd = MV_TABLE.join(DIV_P_TABLE.iloc[:, LAG_NUM * 1:LAG_NUM * 2], how='left').to_numpy('<f8')  # dvd_pre_tax
np_report_period = MV_TABLE.join(DIV_P_TABLE.iloc[:, LAG_NUM * 2:], how='left').to_numpy('<u4')  # report_period

print('end', time.time() - st)
del MV_TABLE, DIV_P_TABLE  # 释放内存

##################################################################
# 3.求出用于计算的不同矩阵
##################################################################
print('start', time.time() - st)
np_info = np_ann_date < np_trade_date
np_dvd = np_dvd * np_info
np_report_period = np_report_period * np_info
np_report_year = (np_report_period // 10000).astype('<u2')
np_ar_factor = np.where(np_report_period != 0, (1 / (np_report_period % 10000 / 1231) - 1), 0)
del np_report_period, np_info
print('ok', time.time() - st)

##################################################################
# 4.在目标输出列中填充
##################################################################
print('start', time.time() - st)
np_lag_year = np.array([np_trade_date[:, 0] // 10000]).T - (np.arange(LAG_PERIOD) + 1)  # 生成目标滞后年份矩阵
np_dvd_lag = np_trade_date[:, 0].reshape(-1, 1)  # 生成储存结构
for i in tqdm(range(LAG_PERIOD)):  # 在目标滞后年份矩阵中滑动填充
    mask_dvd = (np.where(np_report_year == np.tile(np_lag_year[:, i].reshape(-1, 1), LAG_NUM), True, False)
                * np_dvd).sum(axis=1).reshape(-1, 1)  # 累加同一年份的信息
    np_dvd_lag = np.concatenate((np_dvd_lag, mask_dvd), axis=1)  # 拼接目标滞后年份
print('end', time.time() - st)
print('end', time.time() - st)

# np_dv
# 取出t-0滞后 生成比较矩阵
# np_lag_0 = np.array([np_lag_year[:, 0]] * LAG_NUM).T

# 比较
# mask_0 = np.where(np_lag_0 == np_report_year, True, False)
# res_0 = mask_0 * np_dvd
# 累加有用信息
# res_1 = res_0.sum(axis=1).reshape(-1, 1)

# res_1 = np.flipud(np.flip(np.flipud(np.flip(res_0)).sum(axis=1))).reshape(-1, 1)
# np_report_year_2 = np_report_year
# mask = np.where(np_lag_year[:, 3] == np_report_year[:, 3], np_dvd[:, 3], 0).reshape(-1, 1)
# mask1 = np.where(np_report_year == 2018, True, False)


# res2 = mask1 * np_dvd
# res3 = np.flipud(np.flip(np.flipud(np.flip(res2)).cumsum(axis=1)))
# res3 = res2[::-1].cumsum(axis=1)[::-1]
print('ok', time.time() - st)
print('ok', time.time() - st)
df_2 = pd.DataFrame()
for i in tqdm(range(LAG_PERIOD)):  # 目标滞后期输出列 # 每个循环7秒
    df_1['target_ar_{}'.format(i)] = 0  # 年化因子激活矩阵
df_1['target_div_{}'.format(i)] = 0  # 目标分红矩阵
for j in reversed(range(LAG_NUM)):  # 迭代填充 累加报告期到目标日期
    df_1['target_div_{}'.format(i)] = pd.eval(
        'df_1.target_div_{i} + df_1.dvd_info_{j}*(df_1.target_year_{i}==df_1.report_year_{j})'.format(
            i=i, j=j))  # 目标年份矩阵
df_1['target_ar_{}'.format(i)] = pd.eval(
    'df_1.target_ar_{i}*(df_1.target_year_{i}!=df_1.report_year_{j}) + df_1.ar_factor_{j} *(df_1.target_year_{i}==df_1.report_year_{j})'.format(
        i=i, j=j))  # 目标年份矩阵 -年化
df_2['target_div_{}'.format(i)] = df_1['target_div_{}'.format(i)]
df_2['target_div_ar_{}'.format(i)] = pd.eval('df_1.target_div_{i}*(1+df_1.target_ar_{i})'.format(i=i))
del df_1

##################################################################
# 5.预期分红计算 df_div
##################################################################
# ---------------线性回归法---------------#
Y = np.array(df_2[['target_div_{}'.format(i + 1) for i in reversed(range(OBS_J))]]).T
X = np.array([[1] * OBS_J, range(OBS_J)]).T  # 系数矩阵
X_PRE = np.array([[1] * PRE_K, range(OBS_J, OBS_J + PRE_K)]).T  # 待预测期矩阵
Y_PRED = X_PRE.dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)).T  # OLS参数矩阵公式 Beta=(X'Y)/(X'X), Y=BetaX
Y_PRED = np.where(Y_PRED < 0, 0, Y_PRED)  # 清除为0的预测值
df_div = pd.concat(
    [df_t, pd.DataFrame(Y_PRED, index=df_2.index, columns=['EXP_REG_{}'.format(i) for i in range(PRE_K)])], axis=1)
del X, X_PRE, Y, Y_PRED
# ---------------平均法 历史真实值---------------#
df_div['EXP_AVG'] = np.average(df_2[['target_div_{}'.format(i + 1) for i in reversed(range(OBS_J))]], axis=1)
# ---------------年化法+滞后法---------------# t-0为0时,取t-1年的分红,还为0时取t-2的年化分红
df_div['EXP_AR'] = pd.eval(
    '(df_2.target_div_ar_0>0)*df_2.target_div_ar_0+(df_2.target_div_ar_0<=0)*df_2.target_div_ar_1+(df_2.target_div_ar_1<=0)*df_2.target_div_ar_2')
df_div['EXP_REAL'] = pd.eval(
    '(df_2.target_div_0>0)*df_2.target_div_0+(df_2.target_div_0<=0)*df_2.target_div_1+(df_2.target_div_1<=0)*df_2.target_div_2')
# ---------------实际值---------------#
df_div['REAL'] = df_2['target_div_0']
del df_2

##################################################################
# 6.支付率计算 df_pro
##################################################################
df_pay = pd.merge(df_t, pd.read_feather('data.f'), how='left', on=['stockcode', 'ann_date']).astype(
    {'stockcode': 'category'})
df_pay.index = df_t.index
# ---------------用t-1期不同的分红计算支付率---------------#
df_pay['REAL_RATIO'] = pd.eval('df_div.REAL/df_pay.net_profit_parent_comp_ttm').astype('float16')
df_pay['EXP_REAL_RATIO'] = pd.eval('df_div.EXP_REAL/df_pay.net_profit_parent_comp_ttm').astype('float16')
df_pay['EXP_AVG_RATIO'] = pd.eval('df_div.EXP_AVG/df_pay.net_profit_parent_comp_ttm').astype('float16')
df_pay['EXP_AR_RATIO'] = pd.eval('df_div.EXP_AR/df_pay.net_profit_parent_comp_ttm').astype('float16')
df_pay['EXP_REG_0_RATIO'] = pd.eval('df_div.EXP_REG_0/df_pay.net_profit_parent_comp_ttm').astype('float16')
del df_t, df_pay['net_profit_parent_comp_ttm']
