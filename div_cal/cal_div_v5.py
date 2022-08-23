import time
import numpy as np
import pandas as pd

# ----------------参数和命名----------------#
LAG_PERIOD = 4  # 滞后期

# ----------------读取原始数据----------------#
DIV_TABLE = pd.read_parquet('AShareDividend.parquet')

# ----------------筛选计算列----------------#
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE = DIV_TABLE[['stockcode', 'report_period', 'ann_date', 'cash_dvd_per_sh_pre_tax', 's_div_baseshare']]

##################################################################
# 1.转换分红的面板数据为方便计算的矩阵
##################################################################

# ----------------排序后保留20期最近的历史记录----------------#
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总股息

# 按照stockcode升序后,再按照ann_date降序
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)

# ---------------取排序号的前N个数据----------------#
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_PERIOD).copy()
del DIV_TABLE

# ---------------获得组内日期排序序号----------------#
# 由于已经排序,cumcount值就是日期从近到远的顺序
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()

# ---------------转置:按照信息排序----------------#
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])

##################################################################
# 2.用MV_TABLE表与INFO_TABLE表在stockcode上使用左外连接合并
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE[['ann_date', 'report_period', 'dvd_pre_tax']], how='left', on='stockcode')
TARGET_YEAR = (MV_INFO_TABLE['ann_date'].max() // 10000) - 1  # 参照期
MV_INFO_TABLE.fillna(0, inplace=True)  # 没有merge到的缺失信息用0填充
del MV_TABLE, INFO_TABLE, df_group  # 释放内存

# ---------------测试数据---600738.SH在2018年有3次分红------------#
MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']

##################################################################
# 3.矩阵计算
##################################################################
st = time.time()
# ---------------重命名列名----------------#
MV_INFO_TABLE.columns = [i[0] + '_' + str(+i[1]) if isinstance(i, tuple) else i for i in MV_INFO_TABLE.columns]

# ---------------求出用于计算的基本矩阵:信息矩阵,年化因子矩阵,报告期矩阵,累积分红矩阵,分红因子激活矩阵,目标年份矩阵-----------------#
for i in range(LAG_PERIOD):
    # ---------------可用信息矩阵----------------#
    MV_INFO_TABLE['info_{}'.format(i)] = np.where(MV_INFO_TABLE.eval('ann_date>ann_date_{i}'.format(i=i)), 1, 0)
    # ---------------年化因子矩阵----------------#
    #  %10000:取出月和日 /1231 求年化因子
    MV_INFO_TABLE['info_report_ar_{}'.format(i)] = np.where(
        MV_INFO_TABLE.eval('report_period_{i}!=0'.format(i=i)),
        MV_INFO_TABLE.eval('(1/(report_period_{i} % 10000 / 1231)-1)*info_{i}'.format(i=i)), 0.0)

    # ---------------其他矩阵----------------#
    MV_INFO_TABLE.eval("""
    info_report_year_{i} = report_period_{i} * info_{i} //10000 #报告期矩阵
    dvd_pre_tax_sum_{i} = dvd_pre_tax_{i}*info_{i} #累积分红矩阵
    ar_activate_{i} = info_report_ar_{i} #分红因子激活矩阵
    target_year_{i} = @TARGET_YEAR - @i #目标年份矩阵
    target_year_sum_{i} = 0.0 #目标年份累积矩阵-实际
    target_year_sum_ar_{i} = 0.0 #目标年份累积矩阵-年化
    """.format(i=i), inplace=True)

print('矩阵基础计算完成', time.time() - st)

# ---------------在目标输出矩阵中迭代填充-得到最终需要的值----------------#

for i in range(LAG_PERIOD):
    right = LAG_PERIOD - 1 - i  # LAG_PERIOD 4: 0,1,2,3 ;right 3,2,1,0
    left = right - 1
    # 从右往左累积到最左边的列
    if left >= 0:
        same_expr = MV_INFO_TABLE['info_report_year_{}'.format(left)] == MV_INFO_TABLE[
            'info_report_year_{}'.format(right)]
        # ---------------分红累积矩阵----------------#
        MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(left)] = np.where(
            same_expr,
            MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(left)] + MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(right)],
            MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(left)])

        # ---------------分红因子激活矩阵----------------#
        MV_INFO_TABLE['ar_activate_{}'.format(right)] = np.where(
            same_expr, 0,
            MV_INFO_TABLE[('ar_activate_{}'.format(right))])

        del same_expr

    # ---------------填充目标日期矩阵----------------#
    for j in reversed(range(LAG_PERIOD)):
        # 从同一年使用最新的累计分红 ,并排除0,保证分红更新
        same_expr = (MV_INFO_TABLE['target_year_{}'.format(i)] == MV_INFO_TABLE[
            'info_report_year_{}'.format(j)]) & MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(j)] > 0

        # ---------------填充-实际历史分红----------------#
        MV_INFO_TABLE['target_year_sum_{}'.format(i)] = np.where(
            same_expr,
            MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(j)],
            MV_INFO_TABLE['target_year_sum_{}'.format(i)])

        # # ---------------填充-年化历史分红----------------#
        MV_INFO_TABLE['target_year_sum_ar_{}'.format(i)] = np.where(
            same_expr,
            MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(j)] * (1 + MV_INFO_TABLE['ar_activate_{}'.format(j)]),
            MV_INFO_TABLE['target_year_sum_ar_{}'.format(i)])

        del same_expr
    # print('填充完成', time.time() - st)

    # --------------填充当前交易日的T-1期的列----------------#
    MV_INFO_TABLE.eval("""
        target_year_t_{i} = ann_date//10000-1-{i}
        target_flag_{i}= target_year_0-target_year_t_{i}
        target_real_{i}=0
        target_exp_ar_{i}=0
        """.format(i=i), inplace=True)
    # 更新需要的滞后年份
    MV_INFO_TABLE['target_flag_{}'.format(i)] = np.where(MV_INFO_TABLE['target_flag_{}'.format(i)] > LAG_PERIOD, 0,
                                                         MV_INFO_TABLE['target_flag_{}'.format(i)])

    for j in range(LAG_PERIOD):
        # ---------------填充-实际历史分红----------------#
        MV_INFO_TABLE['target_real_{}'.format(i)] = np.where(MV_INFO_TABLE['target_flag_{}'.format(i)] == j,
                                                             MV_INFO_TABLE['target_year_sum_{}'.format(j)],
                                                             MV_INFO_TABLE['target_real_{}'.format(i)])

        # ---------------填充-年化历史分红----------------#
        MV_INFO_TABLE['target_exp_ar_{}'.format(i)] = np.where(MV_INFO_TABLE['target_flag_{}'.format(i)] == j,
                                                               MV_INFO_TABLE['target_year_sum_ar_{}'.format(j)],
                                                               MV_INFO_TABLE['target_exp_ar_{}'.format(i)])
    # print('求滞后完成', time.time() - st)

print('矩阵填充完成', time.time() - st)

# ---------------输出目标数据---------------#
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
MV_INFO_TABLE = MV_INFO_TABLE[
    ['stockcode', 'ann_date'] +
    ['{}_{}'.format(i, j) for i in ['target_year_t', 'target_real', 'target_exp_ar'] for j in range(LAG_PERIOD)]]

# MV_INFO_TABLE.to_csv('final.csv')
# del MV_INFO_TABLE

# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['ann_date'].astype('str').str[:-4].isin(['2017','2018', '2019'])]
