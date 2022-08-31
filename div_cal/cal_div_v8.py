import time
import numpy as np
import pandas as pd
from tqdm import tqdm

##################################################################
# 参数
##################################################################
LAG_NUM = 20  # 历史信息数量
LAG_PERIOD = 4  # 当前交易日滞后期
OBS_J = 3  # 观测期
PRE_K = 1  # 预测期
assert OBS_J < LAG_PERIOD < LAG_NUM

st = time.time()
print('开始计时')
##################################################################
# 1.转换分红的面板数据 按照股票名存储历史信息
##################################################################
DIV_TABLE = pd.read_parquet('AShareDividend.parquet')  # 读取原始数据
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE = DIV_TABLE[['stockcode', 'report_period', 'ann_date', 'cash_dvd_per_sh_pre_tax', 's_div_baseshare']]  # 筛选计算列
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总分红
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)  # 按照stockcode升序后,再按照ann_date降序
DIV_TABLE = DIV_TABLE.groupby(['stockcode']).head(LAG_NUM)  # 取排序号的前N个数据
DIV_TABLE['ANNDATE_MAX'] = DIV_TABLE.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
DIV_P_TABLE = pd.pivot_table(DIV_TABLE, index=['stockcode'], columns=['ANNDATE_MAX'],
                             values=['ann_date', 'report_period', 'dvd_pre_tax'])  # 转置:按照信息排序后转置
DIV_P_TABLE.columns = [i[0] + '_{}'.format(i[1]) for i in DIV_P_TABLE.columns]  # 重命名列名
DIV_P_TABLE.reset_index(inplace=True)
del DIV_TABLE
##################################################################
# 2.用MV_TABLE表与INFO_TABLE表 df_0
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet', columns=['stockcode', 'ann_date', ])
df_0 = pd.merge(MV_TABLE, DIV_P_TABLE, how='left', on='stockcode')

del MV_TABLE, DIV_P_TABLE  # 释放内存
print('合并完成', time.time() - st)

##################################################################
# 测试数据---600738.SH在2018年有3次分红
##################################################################
# df_0 = df_0[df_0['stockcode'] == '600738.SH']
# df_0.sort_values(by='ann_date', ascending=False, inplace=True)
df_t = df_0[['stockcode', 'ann_date']]

##################################################################
# 3.求出用于计算的不同矩阵 df_1 运算准确已经核对
##################################################################
df_1 = pd.DataFrame()
for i in tqdm(range(LAG_NUM)):  # 每个循环1.5秒
    df_1['report_year_{}'.format(i)] = pd.eval(
        'df_0.report_period_{i}//10000 *(df_0.ann_date_{i}<df_0.ann_date)'.format(i=i))
    df_1['ar_factor_{}'.format(i)] = pd.eval(
        '(1/(df_0.report_period_{i}%10000/1231)-1) *(df_0.ann_date_{i}<df_0.ann_date) '.format(i=i))
    df_1['dvd_info_{}'.format(i)] = pd.eval('df_0.dvd_pre_tax_{i} *(df_0.ann_date_{i}<df_0.ann_date)'.format(i=i))
    # np.Where的速度比pd.fillna快 分开写where比合在一起写快
    df_1['report_year_{}'.format(i)] = np.where(np.isnan(df_1['report_year_{}'.format(i)]), 0,
                                                df_1['report_year_{}'.format(i)])
    df_1['ar_factor_{}'.format(i)] = np.where(np.isnan(df_1['ar_factor_{}'.format(i)]), 0,
                                              df_1['ar_factor_{}'.format(i)])
    df_1['ar_factor_{}'.format(i)] = np.where(np.isinf(df_1['ar_factor_{}'.format(i)]), 0,
                                              df_1['ar_factor_{}'.format(i)])  # isinf修正除0错误
    df_1['dvd_info_{}'.format(i)] = np.where(np.isnan(df_1['dvd_info_{}'.format(i)]), 0,
                                             df_1['dvd_info_{}'.format(i)])
print('基础矩阵计算完成', time.time() - st)

##################################################################
# 4.在目标输出列中填充 df_2
##################################################################
for i in range(LAG_PERIOD):
    df_1['target_year_{}'.format(i)] = pd.eval('df_0.ann_date//10000-1-{i}'.format(i=i))  # 目标年份矩阵
del df_0
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
    df_2['target_div_ar_{}'.format(i)] = pd.eval('df_1.target_div_{i}*(1+df_1.target_ar_{i})'.format(i=i))  # 目标年份矩阵
del df_1
print('目标年份填充完成', time.time() - st)

##################################################################
# 5.预期分红计算 df_t
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

# ---------------年化法+滞后法---------------#  t-0为0时,取t-1年的分红,还为0时取t-2的年化分红
df_div['EXP_AR'] = pd.eval(
    '(df_2.target_div_ar_0>0)*df_2.target_div_ar_0+(df_2.target_div_ar_0<=0)*df_2.target_div_ar_1+(df_2.target_div_ar_1<=0)*df_2.target_div_ar_2')
df_div['EXP_REAL'] = pd.eval(
    '(df_2.target_div_0>0)*df_2.target_div_0+(df_2.target_div_0<=0)*df_2.target_div_1+(df_2.target_div_1<=0)*df_2.target_div_2')
# ---------------实际值---------------#
df_div['REAL'] = df_2['target_div_0']

del df_2
print('预期计算完成', time.time() - st)  # 20次历史信息用时60秒,30次用时80秒

##################################################################
# 1.2 转换利润表的面板数据 按照股票名存储历史信息
##################################################################
# PROFIT_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)
# df_t.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)
df_pro = pd.merge(df_t, pd.read_feather('data.f'), how='left', on=['stockcode', 'ann_date'], )
df_pro.index = df_t.index
df_pro['REAL_RATIO'] = pd.eval('df_div.REAL/df_pro.net_profit_parent_comp_ttm')
print('支付率计算完成', time.time() - st)
print(df_pro.info())

##################################################################
#  调试 #
##################################################################
# fill_columns = [j + '_{}'.format(i) for i in range(LAG_NUM) for j in ['ann_date', 'report_period', 'dvd_pre_tax']]
# for i in range(LAG_PERIOD):
#     df_2['target_ar_{}'.format(i)] = df_1['target_ar_{}'.format(i)]
# for j in range(LAG_NUM):
#     df_2['report_year_{}'.format(j)] = df_1['report_year_{}'.format(j)]
#     df_2['ar_factor_{}'.format(j)] = df_1['ar_factor_{}'.format(j)]
