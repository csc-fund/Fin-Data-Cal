import time
import numpy as np
import pandas as pd

# ----------------参数和命名----------------#
LAG_PERIOD = 5  # 滞后期: 当前时期为T,该参数表示使用了[T-2,T-3...,T-LAG_PERIOD]来预测T-1
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


##################################################################
# 2.MV_TABLE表与INFO_TABLE表进行矩阵计算
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')
MV_TABLE = MV_TABLE.iloc[-10000:, :]
MV_TABLE = MV_TABLE[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE[['ann_date', 'report_period']], how='left', on='stockcode')

# ---------------计算可用历史信息的信息矩阵----------------#
for i in range(LAG_PERIOD):
    MV_INFO_TABLE['INFO_{}'.format(i)] = np.where(MV_INFO_TABLE['ann_date'] > MV_INFO_TABLE[('ann_date', i)], 1, 0)

en = time.time()
per = en - st
