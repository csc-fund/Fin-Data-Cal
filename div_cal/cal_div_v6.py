import time
import numpy as np
import pandas as pd

# ----------------参数----------------#
LAG_NUM = 5  # 历史信息滞后数量
LAG_PERIOD = 4  # 历史信息滞后期
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
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_NUM).copy()  # 取排序号的前N个数据
del DIV_TABLE
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])  # 转置:按照信息排序后转置
INFO_TABLE.columns = [i[0] + '_{}'.format(i[1]) for i in INFO_TABLE.columns]  # 重命名列名
INFO_TABLE.reset_index(inplace=True)

##################################################################
# 2.用MV_TABLE表与INFO_TABLE表在stockcode上使用左外连接合并
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE, how='left', on='stockcode')
MV_INFO_TABLE.fillna(0, inplace=True)  # 没有merge到的缺失信息用0填充
info_dtype = {i: 'uint32' for i in INFO_TABLE.columns if ('ann_date' in i) | ('report_period' in i)}
info_dtype.update({'stockcode': 'category'})
MV_INFO_TABLE = MV_INFO_TABLE.astype(info_dtype)  # 数据压缩
# TARGET_YEAR = (int(MV_INFO_TABLE['ann_date'].max()) // 10000) - 1  # 参照期
del MV_TABLE, INFO_TABLE, df_group  # 释放内存

##################################################################
# 测试数据---600738.SH在2018年有3次分红
##################################################################
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']
st = time.time()

##################################################################
# 3.求出用于计算的不同矩阵
##################################################################
for i in range(LAG_NUM):
    MV_INFO_TABLE.eval("""
    info_{i} = ann_date_{i}<ann_date                                  #可用信息矩阵
    dvd_pre_tax_{i} = dvd_pre_tax_{i} * info_{i}                         #分红矩阵
    report_year_{i} = report_period_{i} * info_{i} // 10000                  #报告期矩阵
    ar_factor_{i} = (1/(report_period_{i}[report_period_{i}!=0]%10000/1231)-1)*info_{i}    #年化因子矩阵 
    ar_factor_{i}=ar_factor_{i}.fillna(0)                                       #年化因子矩阵 nan值处理 
    """.format(i=i), inplace=True)

# pd.eval('MV_INFO_TABLE')
# MV_INFO_TABLE = MV_INFO_TABLE.iloc[:, [1, 2]] + MV_INFO_TABLE.iloc[:, (LAG_PERIOD * 3 + 2):]  # 压缩数据
print('基础矩阵计算完成', time.time() - st)

##################################################################
# 4.在目标输出列中用where迭代填充
##################################################################
# MV_INFO_TABLE.eval('ar_activate_xxx=0', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx= ar_activate_xxx *(target_year_3!=report_year_3)+ ar_factor_3 *(target_year_3==report_year_3)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx= ar_activate_xxx *(target_year_3!=report_year_2)+ ar_factor_2 *(target_year_3==report_year_2)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx= ar_activate_xxx *(target_year_3!=report_year_1)+ ar_factor_1 *(target_year_3==report_year_1)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx=ar_factor_1 *(target_year_3==report_year_1)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx=ar_factor_0 *(target_year_3==report_year_0)', inplace=True)

for i in range(LAG_PERIOD):  # 填充目标日期矩阵
    MV_INFO_TABLE.eval("""
    target_year_{i} = ann_date//10000-1-{i}      #目标年份矩阵
    target_div_{i} = 0                           #目标年份 分红矩阵-实际
    target_div_ar_{i} = 0                           #目标年份 分红矩阵-年化
    target_ar_{i} = 0                            #目标年份 年化因子激活矩阵
    """.format(i=i), inplace=True)
    for j in reversed(range(LAG_PERIOD)):  # 迭代填充 累加报告期到目标日期
        MV_INFO_TABLE.eval("""
        target_div_{i} = target_div_{i} + dvd_pre_tax_{j}*(target_year_{i}==report_year_{j})
        target_ar_{i} = target_ar_{i}*(target_year_{i}!=report_year_{j}) + ar_factor_{j} *(target_year_{i}==report_year_{j})
        target_div_ar_{i} = target_div_{i}*(1+target_ar_{i})
        """.format(i=i, j=j), inplace=True)

print('目标年份填充完成', time.time() - st)
##################################################################
# 调试信息
##################################################################
MV_INFO_TABLE = MV_INFO_TABLE[
    ['stockcode'] + ['ann_date'] +
    ['{}_{}'.format(i, j) for i in ['report_year', 'ar_factor'] for j in range(LAG_NUM)] +
    ['{}_{}'.format(i, j) for i in ['target_year', 'target_ar', 'target_div', 'target_div_ar'] for j in
     range(LAG_PERIOD)]]
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)

print(MV_INFO_TABLE.info())



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
MV_INFO_TABLE = MV_INFO_TABLE[['test'] +
                              ['stockcode', 'ann_date'] + ['EXP_REG_0'] + ['EXP_AVG'] + ['EXP_AR'] + ['EXP_LAG']
                              + ['{}_{}'.format(i, j) for i in ['target_year_t', 'target_exp_real'] for j in
                                 range(LAG_PERIOD)]
                              ]
# df6['id']=pd.concat([df['id'] for df in dfs])
# MV_INFO_TABLE.to_csv('final.csv')
