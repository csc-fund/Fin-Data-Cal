import time
import numpy as np
import pandas as pd

# ----------------参数和命名----------------#
LAG_PERIOD = 10  # 滞后期: 当前时期为T,该参数表示使用了[T-2,T-3...,T-LAG_PERIOD]来预测T-1
REFER_DATE = 'ann_date'  # 分红确认的日期: 可选ann_date,s_div_prelandate
MERGE_COLUMN = ['report_year', 'dvd_pre_tax_sum', REFER_DATE + '_max']  # 计算出的列的命名

# ----------------读取原始数据----------------#

DIV_TABLE = pd.read_parquet('AShareDividend.parquet')

# ----------------筛选计算列----------------#
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE = DIV_TABLE[['stockcode', 'report_period', REFER_DATE, 'cash_dvd_per_sh_pre_tax', 's_div_baseshare']]

##################################################################
# 1.转换分红的面板数据为方便计算的矩阵
##################################################################
st = time.time()
# ----------------排序后保留20期最近的历史记录----------------#
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总股息
# 按照stockcode升序后,再按照ann_date降序
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)

# ---------------取排序号的前N个数据----------------#
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_PERIOD).copy()

# ---------------计算组内日期排序----------------#
# 由于已经排序,cumcount值就是信息日期
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()

# ---------------转置----------------#
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])
# INFO_TABLE.reset_index(inplace=True)

st = time.time()
##################################################################
# 2.MV_TABLE表与INFO_TABLE表进行矩阵计算
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')
# MV_TABLE = MV_TABLE.iloc[-10000:, :]
MV_TABLE = MV_TABLE[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE[['ann_date', 'report_period', 'dvd_pre_tax']], how='left', on='stockcode')

print(time.time()-st)
MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']

# ---------------矩阵运算----------------#
for i in range(LAG_PERIOD):
    # ---------------可用信息矩阵----------------#
    MV_INFO_TABLE[('info', i)] = np.where(MV_INFO_TABLE['ann_date'] > MV_INFO_TABLE[('ann_date', i)], 1, 0)

for i in range(LAG_PERIOD):
    # ---------------可用报告期矩阵----------------#
    MV_INFO_TABLE[('info_report', i)] = MV_INFO_TABLE[('info', i)] * MV_INFO_TABLE[('report_period', i)]
    MV_INFO_TABLE[('info_report', i)].fillna(0.0, inplace=True)

for i in range(LAG_PERIOD):
    # ---------------可用报告期矩阵-取出年份----------------#
    MV_INFO_TABLE[('info_report_year', i)] = MV_INFO_TABLE[('info_report', i)].astype('str').str[:4].astype('float')

for i in range(LAG_PERIOD):
    # ---------------可用报告期矩阵-取出分红类型-取出简单年化系数----------------#
    MV_INFO_TABLE[('info_report_ar', i)] = np.where(
        MV_INFO_TABLE[('info_report', i)] != 0.0,
        MV_INFO_TABLE[('info_report', i)].astype('str').str[4:].str.lstrip('0'), 0.0)

    # 求年化
    MV_INFO_TABLE[('info_report_ar', i)] = MV_INFO_TABLE[('info_report_ar', i)].astype('float') / 1231.0

    # 年化系数
    MV_INFO_TABLE[('info_report_ar', i)] = np.where(
        MV_INFO_TABLE[('info_report_ar', i)] != 0.0,
        (1.0 / MV_INFO_TABLE[('info_report_ar', i)]) - 1.0, 0.0)

for i in range(LAG_PERIOD):
    # ---------------可用报告期矩阵-取出分红类型-年化判断----------------#
    MV_INFO_TABLE[('info_report_isar', i)] = np.where(MV_INFO_TABLE[('info_report_ar', i)] != 0.0, 1, 0)

for i in range(LAG_PERIOD):
    # ---------------可用报告期矩阵-取出分红类型-取出简单年化系数-求出总年化分红----------------#
    MV_INFO_TABLE[('dvd_pre_tax', i)].fillna(0.0, inplace=True)
    # MV_INFO_TABLE[('info_div_ar', i)] = MV_INFO_TABLE[('info_report_ar', i)] * MV_INFO_TABLE[('dvd_pre_tax', i)]

for i in range(LAG_PERIOD):
    # ---------------复制久的isar----------------#
    MV_INFO_TABLE[('info_report_isar_new', i)] = MV_INFO_TABLE[('info_report_isar', i)]

for right in reversed(range(LAG_PERIOD)):
    # ---------------按照合并年份更新年化股息----------------#
    left = right - 1
    if left < 0:
        break
    MV_INFO_TABLE[('info_report_isar_new', right)] = np.where(
        MV_INFO_TABLE[('info_report_year', left)] == MV_INFO_TABLE[('info_report_year', right)],
        0, MV_INFO_TABLE[('info_report_isar', right)])

for i in range(LAG_PERIOD):
    # ---------------复制久的isar----------------#
    MV_INFO_TABLE[('dvd_pre_tax_sum', i)] = MV_INFO_TABLE[('dvd_pre_tax', i)] * MV_INFO_TABLE[('info', i)]

for right in reversed(range(LAG_PERIOD)):
    # ---------------按照合并年份更新年化股息----------------#
    left = right - 1
    if left < 0:
        break
    MV_INFO_TABLE[('dvd_pre_tax_sum', left)] = np.where(
        MV_INFO_TABLE[('info_report_year', left)] == MV_INFO_TABLE[('info_report_year', right)],
        MV_INFO_TABLE[('dvd_pre_tax_sum', left)] + MV_INFO_TABLE[('dvd_pre_tax_sum', right)],
        MV_INFO_TABLE[('dvd_pre_tax_sum', left)])

for i in range(LAG_PERIOD):
    # ---------------复制久的isar----------------#
    MV_INFO_TABLE[('dvd_pre_tax_sum_ar', i)] = MV_INFO_TABLE[('dvd_pre_tax_sum', i)] + MV_INFO_TABLE[
        ('info_report_isar_new', i)] * MV_INFO_TABLE[('dvd_pre_tax_sum', i)] * MV_INFO_TABLE[('info_report_ar', i)]

for i in range(LAG_PERIOD):
    MV_INFO_TABLE[('info_year_flag', i)] = 1 * MV_INFO_TABLE[('info', i)]

for right in reversed(range(LAG_PERIOD)):
    # ---------------求出最新列标志----------------#
    left = right - 1
    if left < 0:
        break
    MV_INFO_TABLE[('info_year_flag', right)] = np.where(
        MV_INFO_TABLE[('info_report_year', left)] == MV_INFO_TABLE[('info_report_year', right)],
        0, 1)

for i in range(LAG_PERIOD):
    # ---------------得到最终的累积年化和---------------#
    MV_INFO_TABLE[('dvd_pre_tax_final', i)] = MV_INFO_TABLE[('info_year_flag', i)] * MV_INFO_TABLE[
        ('dvd_pre_tax_sum_ar', i)]

# ---------------目标年份矩阵----------------#
for i in range(LAG_PERIOD):
    MV_INFO_TABLE[('year', i)] = 2020.0 - i
    # pd.concat([], axis=1)
for i in range(LAG_PERIOD):
    MV_INFO_TABLE[('year_sum', i)] = 0.0

# ---------------在目标年份矩阵中迭代合并-得到最终的总分红----------------#
for i in range(LAG_PERIOD):
    for j in range(LAG_PERIOD):
        MV_INFO_TABLE[('year_sum', i)] = np.where(
            (MV_INFO_TABLE[('year', i)] == MV_INFO_TABLE[('info_report_year', j)]) & (
                    MV_INFO_TABLE[('dvd_pre_tax_final', j)] > 0),
            MV_INFO_TABLE[('dvd_pre_tax_final', j)], MV_INFO_TABLE[('year_sum', i)])

# ---------------测试数据---------------#
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['ann_date'].astype('str').str[:-4].isin(['2017','2018', '2019'])]
en = time.time()
per = en - st
