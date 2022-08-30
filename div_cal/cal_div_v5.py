import time
import numpy as np
import pandas as pd

# ----------------参数----------------#
LAG_NUM = 10  # 历史信息滞后数量
LAG_PERIOD = 10  # 历史信息滞后期
OBS_J = 3  # 观测期
PRE_K = 1  # 预测期

##################################################################
# 1.转换分红的面板数据为方便计算的矩阵
##################################################################
DIV_TABLE = pd.read_parquet('AShareDividend.parquet')  # 读取原始数据
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE = DIV_TABLE[['stockcode', 'report_period', 'ann_date', 'cash_dvd_per_sh_pre_tax', 's_div_baseshare']]  # 筛选计算列
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总分红
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)  # 按照stockcode升序后,再按照ann_date降序
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_PERIOD).copy()  # 取排序号的前N个数据
del DIV_TABLE
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])  # 转置:按照信息排序后转置

##################################################################
# 2.用MV_TABLE表与INFO_TABLE表在stockcode上使用左外连接合并
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE[['ann_date', 'report_period', 'dvd_pre_tax']], how='left', on='stockcode')
MV_INFO_TABLE.fillna(0, inplace=True)  # 没有merge到的缺失信息用0填充
MV_INFO_TABLE.columns = [i[0] + '_' + str(+i[1]) if isinstance(i, tuple) else i for i in MV_INFO_TABLE.columns]  # 重命名列名
TARGET_YEAR = (int(MV_INFO_TABLE['ann_date'].max()) // 10000) - 1  # 参照期
del MV_TABLE, INFO_TABLE, df_group  # 释放内存

##################################################################
# 测试数据---600738.SH在2018年有3次分红
##################################################################
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']
st = time.time()

##################################################################
# 3.求出用于计算的不同矩阵
##################################################################
for i in range(LAG_PERIOD):  # 信息矩阵,年化因子矩阵,报告期矩阵,累积分红矩阵,分红因子激活矩阵,目标年份矩阵
    # ---------------可用信息矩阵----------------#
    MV_INFO_TABLE['info_{}'.format(i)] = np.where(MV_INFO_TABLE.eval('ann_date>ann_date_{i}'.format(i=i)), 1, 0)
    # ---------------年化因子矩阵----------------#
    MV_INFO_TABLE['info_report_ar_{}'.format(i)] = np.where(
        MV_INFO_TABLE.eval('report_period_{i}!=0'.format(i=i)),
        MV_INFO_TABLE.eval('(1/(report_period_{i} % 10000 / 1231)-1)*info_{i}'.format(i=i)), 0.0)
    # ---------------其他矩阵----------------#
    MV_INFO_TABLE.eval("""
       
        info_report_year_{i} = report_period_{i} * info_{i} //10000 #报告期矩阵
        dvd_pre_tax_sum_{i} = dvd_pre_tax_{i} * info_{i} #累积分红矩阵
        ar_activate_{i} = info_report_ar_{i} #分红因子激活矩阵
        target_year_{i} = @TARGET_YEAR - @i #目标年份矩阵
        target_year_cum_{i} = 0.0 #目标年份累积矩阵-实际
        target_year_cum_ar_{i} = 0.0 #目标年份累积矩阵-年化
        """.format(i=i), inplace=True)

print('基础矩阵计算完成', time.time() - st)

##################################################################
# 4.在目标输出列中用where迭代填充
##################################################################

for i in range(LAG_PERIOD):  # 填充目标年份
    right = LAG_PERIOD - 1 - i  # LAG_PERIOD 4: 0,1,2,3 ;right 3,2,1,0
    left = right - 1
    if left >= 0:  # 从右往左累积到最左边的列
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
            MV_INFO_TABLE['ar_activate_{}'.format(right)])
        del same_expr

    # ---------------填充目标日期矩阵----------------#
    for j in reversed(range(LAG_PERIOD)):
        # 从同一年使用最新的累计分红 ,并排除0,保证分红更新
        same_expr = (MV_INFO_TABLE['target_year_{}'.format(i)] == MV_INFO_TABLE[
            'info_report_year_{}'.format(j)]) & MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(j)] > 0
        # same_expr = MV_INFO_TABLE.eval('(target_year_{i}==info_report_year_{j})&dvd_pre_tax_sum_{j}>0'.format(i=i, j=j))
        # ---------------填充-实际历史分红----------------#
        MV_INFO_TABLE['target_year_cum_{}'.format(i)] = np.where(
            same_expr,
            MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(j)],
            MV_INFO_TABLE['target_year_cum_{}'.format(i)])

        # # ---------------填充-年化历史分红----------------#
        MV_INFO_TABLE['target_year_cum_ar_{}'.format(i)] = np.where(
            same_expr,
            MV_INFO_TABLE['dvd_pre_tax_sum_{}'.format(j)] * (1 + MV_INFO_TABLE['ar_activate_{}'.format(j)]),
            MV_INFO_TABLE['target_year_cum_ar_{}'.format(i)])

        del same_expr

print('目标年份填充完成', time.time() - st)

for i in range(LAG_PERIOD):  # 填充滞后年份
    # --------------填充当前交易日的T-1期的列----------------#
    MV_INFO_TABLE.eval("""
                target_year_t_{i} = ann_date//10000-1-{i}
                target_flag_{i}= target_year_0-target_year_t_{i}
                target_exp_real_{i}=0
                target_exp_ar_{i}=0
                """.format(i=i), inplace=True)
    # 更新需要的滞后年份
    MV_INFO_TABLE['target_flag_{}'.format(i)] = np.where(MV_INFO_TABLE['target_flag_{}'.format(i)] > LAG_PERIOD - 1, 0,
                                                         MV_INFO_TABLE['target_flag_{}'.format(i)])

    for j in range(LAG_PERIOD):
        same_expr = MV_INFO_TABLE['target_flag_{}'.format(i)] == j
        # ---------------填充-实际历史分红----------------#
        MV_INFO_TABLE['target_exp_real_{}'.format(i)] = np.where(
            same_expr,
            MV_INFO_TABLE['target_year_cum_{}'.format(j)],
            MV_INFO_TABLE['target_exp_real_{}'.format(i)])

        # # ---------------填充-年化历史分红----------------#
        MV_INFO_TABLE['target_exp_ar_{}'.format(i)] = np.where(
            same_expr,
            MV_INFO_TABLE['target_year_cum_ar_{}'.format(j)],
            MV_INFO_TABLE['target_exp_ar_{}'.format(i)])
        del same_expr

print('滞后年份填充完成', time.time() - st)

##################################################################
# 5.预期分红计算
##################################################################
# ---------------线性回归法---------------#
# print(['target_exp_real_{}'.format(i + 1) for i in reversed(range(OBS_J))])
# Y=MV_INFO_TABLE.columns[['target_exp_real_{}'.format(i + 1) for i in reversed(range(OBS_J))]]
Y = np.array(MV_INFO_TABLE[['target_exp_real_{}'.format(i + 1) for i in reversed(range(OBS_J))]]).T
print(Y.shape)
X = np.array([[1] * OBS_J, range(OBS_J)]).T  # 系数矩阵
X_PRE = np.array([[1] * PRE_K, range(OBS_J, OBS_J + PRE_K)]).T  # 待预测期矩阵
Y_PRED = X_PRE.dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)).T  # OLS参数矩阵公式 Beta=(X'Y)/(X'X), Y=BetaX
Y_PRED = np.where(Y_PRED < 0, 0, Y_PRED)  # 清除为0的预测值
MV_INFO_TABLE = pd.concat(
    [MV_INFO_TABLE,
     pd.DataFrame(Y_PRED, index=MV_INFO_TABLE.index, columns=['EXP_REG_{}'.format(i) for i in range(PRE_K)])], axis=1)
del X, X_PRE, Y, Y_PRED

# ---------------平均法 历史真实值---------------#
MV_INFO_TABLE['EXP_AVG'] = np.average(
    MV_INFO_TABLE[['target_exp_real_{}'.format(i + 1) for i in reversed(range(OBS_J))]], axis=1)

# ---------------年化法---------------#
MV_INFO_TABLE['EXP_AR'] = np.where(MV_INFO_TABLE['target_exp_ar_0'] > 0, MV_INFO_TABLE['target_exp_ar_0'],
                                   MV_INFO_TABLE['target_exp_ar_1'])  # 年化值为0的时候取t-1年的年化分红
MV_INFO_TABLE['EXP_AR'] = np.where(MV_INFO_TABLE['target_exp_ar_0'] > 0, MV_INFO_TABLE['target_exp_ar_0'],
                                   MV_INFO_TABLE['target_exp_ar_2'])  # t-1年化值为0的时候取t-2的年化分红

# ---------------滞后法 取上一年实际分红---------------#
MV_INFO_TABLE['EXP_LAG'] = np.where(MV_INFO_TABLE['target_exp_real_0'] > 0, MV_INFO_TABLE['target_exp_real_0'],
                                    MV_INFO_TABLE['target_exp_real_1'])  # 实际值为0的时候取t-1年的实际分红
MV_INFO_TABLE['EXP_LAG'] = np.where(MV_INFO_TABLE['target_exp_real_0'] > 0, MV_INFO_TABLE['target_exp_real_0'],
                                    MV_INFO_TABLE['target_exp_real_2'])  # 实际值为0的时候取t-1年的实际分红

print('预期计算完成', time.time() - st)
##################################################################
# 输出目标数据
##################################################################
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
MV_INFO_TABLE = MV_INFO_TABLE[['test']+
    ['stockcode', 'ann_date'] + ['EXP_REG_0'] + ['EXP_AVG'] + ['EXP_AR'] + ['EXP_LAG']
    + ['{}_{}'.format(i, j) for i in ['target_year_t', 'target_exp_real'] for j in range(LAG_PERIOD)]
    ]
# df6['id']=pd.concat([df['id'] for df in dfs])
# MV_INFO_TABLE.to_csv('final.csv')
