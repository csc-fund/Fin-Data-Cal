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
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_NUM).copy()  # 取排序号的前N个数据
del DIV_TABLE
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])  # 转置:按照信息排序后转置
INFO_TABLE.columns = [i[0] + '_{}'.format(i[1]) for i in INFO_TABLE.columns]  # 重命名列名
INFO_TABLE.reset_index(inplace=True)

##################################################################
# 2.用MV_TABLE表与INFO_TABLE表 在stockcode上使用左外连接合并
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet', columns=['stockcode', 'ann_date', ])
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE, how='left', on='stockcode')
del MV_TABLE, INFO_TABLE, df_group  # 释放内存
print('合并完成', time.time() - st)

##################################################################
# 测试数据---600738.SH在2018年有3次分红
##################################################################
MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']

##################################################################
# 3.求出用于计算的不同矩阵
##################################################################
fill_columns = [j + '_{}'.format(i) for i in range(LAG_NUM) for j in ['ann_date', 'report_period', 'dvd_pre_tax']]
MV_INFO_TABLE.loc[:, fill_columns].fillna(0, inplace=True)
print('fillna完成', time.time() - st)
df_o = MV_INFO_TABLE
df_new = pd.DataFrame()
for i in tqdm(range(LAG_NUM)):
    eval_str = ['df_o.dvd_pre_tax_{i} *(df_o.ann_date_{i}<df_o.ann_date) #分红矩阵',
                'df_o.report_period_{i}//10000 *(df_o.ann_date_{i}<df_o.ann_date) #报告期矩阵',
                '(1/(df_o.report_period_{i}%10000/1231)-1) *(df_o.ann_date_{i}<df_o.ann_date)  #年化因子矩阵']
    # tar_column = ['dvd_info_{i}', 'report_year_{i}', 'ar_factor_{i}']

    df_new['dvd_info_{}'.format(i)], df_new['report_year_{}'.format(i)], df_new['ar_factor_{}'.format(i)] = map(
        lambda x: pd.eval(str(x).format(i=i)), eval_str)

    #
    # df_new['dvd_info_{}'.format(i)] = pd.eval(eval_str[0].format(i=i))
    # df_new['report_year_{}'.format(i)] = pd.eval(eval_str[0].format(i=i))
    # df_new['ar_factor_{}'.format(i)] = pd.eval(eval_str[0].format(i=i))

    # df_new['dvd_info_{}'.format(i)] = pd.eval(eval_str1)
    # for s in eval_str:
    #     print(s.format(i=i))
    #     pd.eval(str(s).format(i=i))
    # pd.eval("""
    # df_n.dvd_info_{i} = df_o.dvd_pre_tax_{i} *(df_o.ann_date_{i}<df_o.ann_date)                          #分红矩阵
    # df_n.report_year_{i} = df_o.report_period_{i}//10000 *(df_o.ann_date_{i}<df_o.ann_date)              #报告期矩阵
    # df_n.ar_factor_{i} = (1/(df_o.report_period_{i}%10000/1231)-1) *(df_o.ann_date_{i}<df_o.ann_date)    #年化因子矩阵
    # """.format(i=i), inplace=True)
    df_new['ar_factor_{}'.format(i)] = np.where(np.isinf(df_new['ar_factor_{}'.format(i)]), 0,
                                                df_new['ar_factor_{}'.format(i)])  # 修正除0错误
del df_o
MV_INFO_TABLE = df_new
print('基础矩阵计算完成', time.time() - st)
exit()
##################################################################
# 4.在目标输出列中填充
##################################################################
for i in tqdm(range(LAG_PERIOD)):
    MV_INFO_TABLE.eval("""
    target_year_{i} = ann_date//10000-1-{i}      #目标年份矩阵
    target_div_{i} = 0                           #目标年份 分红矩阵-实际
    target_ar_{i} = 0                            #目标年份 年化因子激活矩阵
    """.format(i=i), inplace=True)
    for j in reversed(range(LAG_NUM)):  # 迭代填充 累加报告期到目标日期
        MV_INFO_TABLE.eval("""
        target_div_{i} = target_div_{i} + dvd_info_{j}*(target_year_{i}==report_year_{j})
        target_ar_{i} = target_ar_{i}*(target_year_{i}!=report_year_{j}) + ar_factor_{j} *(target_year_{i}==report_year_{j})
        target_div_ar_{i} = target_div_{i}*(1+target_ar_{i}) #年化分红
        """.format(i=i, j=j), inplace=True)
drop_columns = [j + '_{}'.format(i) for i in range(LAG_PERIOD) for j in ['target_year', 'target_ar']]
MV_INFO_TABLE.drop(drop_columns, axis=1, inplace=True)
print('目标年份填充完成', time.time() - st)

##################################################################
# 压缩数据
##################################################################
zip_dtype = {i: 'float32' for i in
             [j + '_{}'.format(i) for i in range(LAG_PERIOD) for j in ['target_div', 'target_div_ar']]}
zip_dtype.update({'stockcode': 'category', 'ann_date': 'uint32'})
MV_INFO_TABLE = MV_INFO_TABLE.astype(zip_dtype)
# MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
print('数据压缩完成', time.time() - st)

##################################################################
# 5.预期分红计算
##################################################################
# ---------------线性回归法---------------#
Y = np.array(MV_INFO_TABLE[['target_div_{}'.format(i + 1) for i in reversed(range(OBS_J))]]).T
X = np.array([[1] * OBS_J, range(OBS_J)]).T  # 系数矩阵
X_PRE = np.array([[1] * PRE_K, range(OBS_J, OBS_J + PRE_K)]).T  # 待预测期矩阵
Y_PRED = X_PRE.dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)).T  # OLS参数矩阵公式 Beta=(X'Y)/(X'X), Y=BetaX
del X, X_PRE, Y
Y_PRED = np.where(Y_PRED < 0, 0, Y_PRED)  # 清除为0的预测值
MV_INFO_TABLE = pd.concat(
    [MV_INFO_TABLE,
     pd.DataFrame(Y_PRED, index=MV_INFO_TABLE.index, columns=['EXP_REG_{}'.format(i) for i in range(PRE_K)])], axis=1)
del Y_PRED

# ---------------平均法 历史真实值---------------#
MV_INFO_TABLE['EXP_AVG'] = np.average(
    MV_INFO_TABLE[['target_div_{}'.format(i + 1) for i in reversed(range(OBS_J))]], axis=1)

# ---------------年化法+滞后法---------------#  t-0为0时,取t-1年的分红,还为0时取t-2的年化分红
MV_INFO_TABLE.eval("""
EXP_AR=(target_div_ar_0>0)*target_div_ar_0+(target_div_ar_0<=0)*target_div_ar_1+(target_div_ar_1<=0)*target_div_ar_2
EXP_REAL=(target_div_0>0)*target_div_0+(target_div_0<=0)*target_div_1+(target_div_1<=0)*target_div_2
""", inplace=True)
print('预期计算完成', time.time() - st)

##################################################################
# 输出目标数据
##################################################################
MV_INFO_TABLE = MV_INFO_TABLE[['stockcode', 'ann_date'] + ['EXP_REAL', 'EXP_AR', 'EXP_AVG', 'EXP_REG_0']]
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
# MV_INFO_TABLE.to_csv('final.csv')


# 生成新列效率
# df6['id']=pd.concat([df['id'] for df in dfs])

# 逻辑表达式
# MV_INFO_TABLE.eval('ar_activate_xxx=0', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx= ar_activate_xxx *(target_year_3!=report_year_3)+ ar_factor_3 *(target_year_3==report_year_3)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx= ar_activate_xxx *(target_year_3!=report_year_2)+ ar_factor_2 *(target_year_3==report_year_2)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx= ar_activate_xxx *(target_year_3!=report_year_1)+ ar_factor_1 *(target_year_3==report_year_1)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx=ar_factor_1 *(target_year_3==report_year_1)', inplace=True)
# MV_INFO_TABLE.eval('ar_activate_xxx=ar_factor_0 *(target_year_3==report_year_0)', inplace=True)
