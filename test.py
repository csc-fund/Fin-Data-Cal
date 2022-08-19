#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2022/8/19 14:23
# @Author    :Colin
# @Note      :None

from functools import reduce

class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for num in nums:
            a = a ^ num
        return a
#     return reduce(lambda x, y: x | y, num)
