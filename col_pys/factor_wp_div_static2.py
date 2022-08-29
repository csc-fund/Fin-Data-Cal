# -*- coding:utf-8 _*-



'''
author = 'wp'
factor_name = 'lyr_div'
L1_category = '风格因子'
L2_category = '红利'
reference = '<<股息率因子全解析>>2018'
description = '静态分红， 过去一年’
expression = ''
datasource = 'wind'
dependency = []
commets = '静态股息率：上年总分红/当前总市值(软件一般按上年的12.31日算市值)'
'''

from dataloader import FinContext
from finlib import  mkt_neutral
from utils import StkBacktest
# from BBData import D
from csclib import rpt_npreperiods_vals
import pandas as pd
import numpy as np
from datetime import date
from csclib import ann2trade
from tqdm import tqdm

from scipy import stats
# stats.linregress()

pd.options.display.max_columns = 999


def static_div(div_records, lag = 2):
    ''' 两年平均 '''
    arr = div_records.values
    end = np.where(~np.isnan(arr))[0][-1]+1
    start = max(0, end-lag)
    return np.mean(arr[start:end])

def stkfunc(df):
    ''' 每个ann_date, 计算上一年的分红总和 '''
    stockcode = df.stockcode.unique()
    ann_date = 1
    update_info = {}
    return stockcode, ann_date, update_info

def generate(ctx):
    div_years = np.arange(2005,date.today().year+4)
    div = ctx.AShareDividend.get_df()
    # eod = ctx.AShareEODPrices.get_df()
    # 只选实施类型的公告 s_div_progress = 3; 去掉94753数据中的 60%, 剩余3000多个数据， 在类型1,2中也可以有分红
    # df = df[df.s_div_progress=='3']
    div = div[div.ann_date>=20070101]
    div = div[div.cash_dvd_per_sh_pre_tax > 0]
    # div = div[div['s_div_object']=='普通股股东']
    # s_div_baseshare为股本基准日期s_div_basedate的股本
    div['cash_div'] = div['cash_dvd_per_sh_pre_tax']*div['s_div_baseshare']
    # df['s_div_object'].unique(): ['普通股股东', 'A股流通股', '重整管理人', '国家股股东', 'A股限售股', 'nan', '公司股东'],
    df = div[['stockcode' ,'ann_date','report_period', 'cash_div']]
    df = df.assign(report_year = (df.report_period/10000).astype(int))
    
    result = []
    for stockcode, sub_df in tqdm(df.groupby('stockcode')):
        ''' 000062.SZ: 都是nan； 000061.SZ：没有递增 ； '000006.SZ'全是nan； '000009.SZ'中途不分'''
        # if stockcode!= '000061.SZ':
        #     continue
        # print(sub_df)
        div_records = pd.Series(np.nan, index=div_years)
        nov_check = list((div_years + 1)*10000 + 1101)   #  11月检查上年分红
        sub_df.drop_duplicates(subset = ['report_period'], keep = 'first', inplace = True)
        sub_df = sub_df.drop_duplicates(subset=['stockcode', 'ann_date', 'report_period'], keep='first').sort_values('ann_date')
        
        sub_df['cum_div'] = sub_df.groupby('report_year')['cash_div'].cumsum()
        sub_df['adj_factor'] = ctx.pct_days(sub_df['report_period'])
        sub_df['adj_cum_div'] = sub_df['cum_div']/sub_df['adj_factor']
        sub_df = sub_df [['stockcode', 'ann_date','report_period','report_year','adj_cum_div','cum_div']]
        for i, (idx , row) in enumerate(sub_df.iterrows()):
            while row.ann_date>=nov_check[0]:
                ann_date = nov_check.pop(0)
                confirmed_year = int(ann_date/10000-1)
                confirmed_year_df = sub_df[sub_df.report_year == confirmed_year]
                if len(confirmed_year_df) == 0:
                    div_records[confirmed_year] = 0.0
                else:
                    div_records[confirmed_year] = sub_df[sub_df.report_year == confirmed_year].iloc[-1].cum_div
                result.append((stockcode, ann_date, static_div(div_records, 2)))
            year = row.report_year
            div_records[year] = row.adj_cum_div
            result.append((stockcode, row.ann_date, static_div(div_records, 2)))
    factor = pd.DataFrame(result, columns = ['stockcode','ann_date','lyr_div'])
    factor['ann_date'] = ann2trade(factor['ann_date'],ctx.trade_dts)
    factor.rename(columns={'ann_date':'trade_dt'}, inplace= True)
    return factor



if __name__ == '__main__':
    repo = r'C:\tmp\qtData'
    ctx = FinContext(repo)
    factor = generate(ctx)
    ctx.save_data('factor_wp_div_static2')
    ctx.load_data('')
    engine = StkBacktest(ctx)
    # mat_alf = engine.df2mat(alf)
    engine.calc_alf(factor, univ='all')

