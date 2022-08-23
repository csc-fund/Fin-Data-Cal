# 下面列出list的几种常用内置方法以及list的几种构建方法

# coding:utf-8

# 导入timeit模块

import timeit

# t1，以+的方式构建列表

'''
	li1 = [1,2]

	li2 = [23,24]

	li = li1 + li2

	#t2，列表生成器

	li = [i for i in range(10000)]

	#t3，将可迭代对象(range)直接转换成列表

	li = list(range(10000))

	#t4，先创建一个空列表，然后用.append方法添加元素

	li = []

	for i in range(10000):
		li.append(i)
'''


# 下面开始测算

# append方法对空列表添加元素构造列表
def t1():
    li = []
    for i in range(10000):
        li.append(i)


# +的方法构造列表
def t2():
    li = []
    for i in range(10000):
        li += [i]


# 列表生成器
def t3():
    li = [i for i in range(10000)]


# 转换可迭代对象为列表
def t4():
    li = list(range(10000))


timer1 = timeit.Timer('t1()', )
print('+:', timer1.timeit(1000))

timer2 = timeit.Timer('t2()', 'from __main__ import t2')
print('append:', timer2.timeit(1000))

timer3 = timeit.Timer('t3()', 'from __main__ import t3')
print('列表生成器:', timer3.timeit(1000))

timer4 = timeit.Timer('t4', 'from __main__ import t4')
print('直接转换可迭代对象:', timer4.timeit(1000))