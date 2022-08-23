import timeit

timer1 = timeit.Timer(stmt="from div_cal import cal_div_v5")
print('运行速度:', timer1.timeit(2))
# print('运行速度:', timeit.timeit(stmt="from div_cal import cal_div_v5", number=10))
