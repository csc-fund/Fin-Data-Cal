import time
import numpy as np
import pandas as pd

# ----------------参数和命名----------------#
TARGET_YEAR = 2020  # 参照期
LAG_PERIOD = 5  # 滞后期
REFER_DATE = 'ann_date'  # 分红确认的日期: 可选ann_date,s_div_prelandate

# ----------------读取原始数据----------------#
DIV_TABLE = pd.read_parquet('AShareDividend.parquet')

# ----------------筛选计算列----------------#
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE = DIV_TABLE[['stockcode', 'report_period', REFER_DATE, 'cash_dvd_per_sh_pre_tax', 's_div_baseshare']]

##################################################################
# 1.转换分红的面板数据为方便计算的矩阵
##################################################################

# ----------------排序后保留20期最近的历史记录----------------#
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总股息

# 按照stockcode升序后,再按照ann_date降序
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)

# ---------------取排序号的前N个数据----------------#
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_PERIOD).copy()

# ---------------计算组内日期排序----------------#
# 由于已经排序,cumcount值就是日期从近到远的顺序
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()

# ---------------转置----------------#
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])

##################################################################
# 2.用MV_TABLE表与INFO_TABLE表在stockcode上使用左连接合并
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')
MV_TABLE = MV_TABLE[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE[['ann_date', 'report_period', 'dvd_pre_tax']], how='left', on='stockcode')
MV_INFO_TABLE.fillna(0, inplace=True)  # 没有merge到的缺失信息用0填充
del MV_TABLE, INFO_TABLE, DIV_TABLE, df_group  # 释放内存

# ---------------测试数据---6007383在2018年3次分红------------#
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']

st = time.time()
##################################################################
# 矩阵计算
##################################################################
for i in range(LAG_PERIOD):
    # ---------------可用信息矩阵----------------#
    MV_INFO_TABLE[('info', i)] = np.where(MV_INFO_TABLE['ann_date'] > MV_INFO_TABLE[('ann_date', i)], 1, 0)

    # ---------------可用报告期矩阵-info_report_year---------------#
    MV_INFO_TABLE[('report_period', i)] = MV_INFO_TABLE[('report_period', i)].astype('int') * MV_INFO_TABLE[('info', i)]
    MV_INFO_TABLE[('info_report_year', i)] = (MV_INFO_TABLE[('report_period', i)] // 10000)  # 取出年

    # ---------------年化因子矩阵----------------#
    #  %10000:取出月和日
    #  /1231 求年化因子
    MV_INFO_TABLE[('info_report_ar', i)] = np.where(
        MV_INFO_TABLE[('report_period', i)] != 0,
        ((1 / ((MV_INFO_TABLE[('report_period', i)] % 10000) / 1231)) - 1.0) * MV_INFO_TABLE[('info', i)], 0.0)

    # ---------------可用累积分红矩阵----------------#
    MV_INFO_TABLE[('dvd_pre_tax_sum', i)] = MV_INFO_TABLE[('dvd_pre_tax', i)] * MV_INFO_TABLE[('info', i)]

    # ---------------分红因子激活矩阵----------------#
    MV_INFO_TABLE[('ar_activate', i)] = MV_INFO_TABLE[('info_report_year', i)]

    # ---------------目标年份矩阵----------------#
    MV_INFO_TABLE[('target_year', i)] = TARGET_YEAR - i
    MV_INFO_TABLE[('target_year_sum', i)] = 0.0
    MV_INFO_TABLE[('target_year_sum_ar', i)] = 0.0

# ---------------在目标年份矩阵中迭代合并-得到最终的总分红----------------#
for i in range(LAG_PERIOD):

    right = LAG_PERIOD - 1 - i  # LAG_PERIOD 4: 0,1,2,3 ;right 3,2,1,0
    left = right - 1
    # 从右往左累积到最左边的列
    if left >= 0:
        # ---------------分红累积矩阵----------------#
        MV_INFO_TABLE[('dvd_pre_tax_sum', left)] = np.where(
            MV_INFO_TABLE[('info_report_year', left)] == MV_INFO_TABLE[('info_report_year', right)],  # 只累加相同年份
            MV_INFO_TABLE[('dvd_pre_tax_sum', left)] + MV_INFO_TABLE[('dvd_pre_tax_sum', right)],
            MV_INFO_TABLE[('dvd_pre_tax_sum', left)])

        # ---------------分红因子激活矩阵----------------#
        MV_INFO_TABLE[('ar_activate', right)] = np.where(
            MV_INFO_TABLE[('info_report_year', right)] == MV_INFO_TABLE[('info_report_year', left)],  # 右边与左边相同时把右边置0
            0, 1)

    # ---------------填充目标日期矩阵----------------#
    for j in reversed(range(LAG_PERIOD)):  # 从同一年使用最新的累计分红 ,并排除0,保证分红更新

        # ---------------填充-实际历史分红----------------#
        MV_INFO_TABLE[('target_year_sum', i)] = np.where(
            (MV_INFO_TABLE[('target_year', i)] == MV_INFO_TABLE[('info_report_year', j)])
            & (MV_INFO_TABLE[('dvd_pre_tax_sum', j)] > 0), MV_INFO_TABLE[('dvd_pre_tax_sum', j)],
            MV_INFO_TABLE[('target_year_sum', i)])

        # ---------------填充-年化历史分红----------------#
        MV_INFO_TABLE[('target_year_sum_ar', i)] = np.where(
            (MV_INFO_TABLE[('target_year', i)] == MV_INFO_TABLE[('info_report_year', j)])
            & (MV_INFO_TABLE[('dvd_pre_tax_sum', j)] > 0),
            MV_INFO_TABLE[('dvd_pre_tax_sum', j)] * (
                        1 + (MV_INFO_TABLE[('info_report_ar', j)] * MV_INFO_TABLE[('ar_activate', j)])),
            MV_INFO_TABLE[('target_year_sum_ar', i)])

# ---------------测试数据---------------#
print(time.time() - st)
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['ann_date'].astype('str').str[:-4].isin(['2017','2018', '2019'])]
