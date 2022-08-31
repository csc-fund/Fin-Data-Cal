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
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)  # 按照stockcode升序后,再按照ann_date降序
DIV_TABLE = DIV_TABLE.groupby(['stockcode']).head(LAG_NUM)  # 取每个股票最近N个历史数据
DIV_TABLE['ANNDATE_MAX'] = DIV_TABLE.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
DIV_P_TABLE = pd.pivot_table(DIV_TABLE, index=['stockcode'], columns=['ANNDATE_MAX'],
                             values=['ann_date', 'report_period', 'dvd_pre_tax'])  # 转置:按照信息排序后转置
DIV_P_TABLE.columns = [i[0] + '_{}'.format(i[1]) for i in DIV_P_TABLE.columns]  # 重命名列名
DIV_P_TABLE.reset_index(inplace=True)
del DIV_TABLE
##################################################################
# 2.合并MV_TABLE表与DIV_P_TABLE表 df_0
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet', columns=['stockcode', 'ann_date', ]).astype({'ann_date': 'uint32'})
df_0 = pd.merge(MV_TABLE, DIV_P_TABLE, how='left', on='stockcode')
del MV_TABLE, DIV_P_TABLE  # 释放内存

##################################################################
# 测试数据---600738.SH在2018年有3次分红
##################################################################
# df_0 = df_0[df_0['stockcode'] == '600738.SH']
# df_0.sort_values(by='ann_date', ascending=False, inplace=True)
df_t = df_0[['stockcode', 'ann_date']].astype({'stockcode': 'category'})  # 前两列标识列

##################################################################
# 3.求出用于计算的不同矩阵 df_1
##################################################################
st = time.time()
df_1 = pd.DataFrame()
for i in tqdm(range(LAG_NUM)):  # 每个循环1.5秒
    df_1['report_year_{}'.format(i)] = pd.eval(
        'df_0.report_period_{i}//10000 *(df_0.ann_date_{i}<df_0.ann_date)'.format(i=i))
    df_1['ar_factor_{}'.format(i)] = pd.eval(
        '(1/(df_0.report_period_{i}%10000/1231)-1) *(df_0.ann_date_{i}<df_0.ann_date) '.format(i=i))
    df_1['dvd_info_{}'.format(i)] = pd.eval('df_0.dvd_pre_tax_{i} *(df_0.ann_date_{i}<df_0.ann_date)'.format(i=i))
    # np.Where的速度比pd.fillna快 分开写where比合在一起写快
    df_1['report_year_{}'.format(i)] = np.where(np.isnan(df_1['report_year_{}'.format(i)]), 0,
                                                df_1['report_year_{}'.format(i)]).astype('uint16')
    df_1['ar_factor_{}'.format(i)] = np.where(np.isnan(df_1['ar_factor_{}'.format(i)]), 0,
                                              df_1['ar_factor_{}'.format(i)])
    df_1['ar_factor_{}'.format(i)] = np.where(np.isinf(df_1['ar_factor_{}'.format(i)]), 0,
                                              df_1['ar_factor_{}'.format(i)]).astype('float16')  # isinf修正除0错误
    df_1['dvd_info_{}'.format(i)] = np.where(np.isnan(df_1['dvd_info_{}'.format(i)]), 0,
                                             df_1['dvd_info_{}'.format(i)])
print('cal1', time.time() - st)
##################################################################
# 4.在目标输出列中填充 df_2
##################################################################
df_2 = pd.DataFrame()
for i in range(LAG_PERIOD):
    df_1['target_year_{}'.format(i)] = pd.eval('df_0.ann_date//10000-1-{i}'.format(i=i)).astype('uint16')  # 目标年份矩阵
del df_0
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
print('cal2', time.time() - st)

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
print('cal3', time.time() - st)
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
print('cal4', time.time() - st)

print(df_div.info(), df_pay.info())
##################################################################
#  调试 #
##################################################################
# fill_columns = [j + '_{}'.format(i) for i in range(LAG_NUM) for j in ['ann_date', 'report_period', 'dvd_pre_tax']]
# for i in range(LAG_PERIOD):
#     df_2['target_ar_{}'.format(i)] = df_1['target_ar_{}'.format(i)]
# for j in range(LAG_NUM):
#     df_2['report_year_{}'.format(j)] = df_1['report_year_{}'.format(j)]
#     df_2['ar_factor_{}'.format(j)] = df_1['ar_factor_{}'.format(j)]
