#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午4:38
# @Author  : ACHIEVE_DREAM
# @File    : simulated_annealing.py
# @Software: Pycharm
from typing import Callable, Any

import numpy as np


def sa(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int = 100,
    initial_temperature: float = 1000,
    cooling_rate: float = 0.99,
    seed: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟退火算法
    :param fitness_func: 待求解问题的优化函数
    :param ndim: 问题维度
    :param bounds: 求解结果的边界范围
    :param max_iters: 迭代次数
    :param initial_temperature: 初始温度
    :param cooling_rate: 温度的冷却速度
    :param seed: 随机数种子
    :return: 最佳解决方案, 最小适应度, 历史适应度
    https://blog.csdn.net/myf_666/article/details/130996611
    """
    # 随机初始化
    rng = np.random.default_rng(seed)
    X = rng.uniform(*bounds, size=(1, ndim))
    best_solution = X.copy()
    ps = kwargs.pop("population_size")

    best_fitness = fitness_func(best_solution, **kwargs)
    current_temperature = initial_temperature
    max_iters = max_iters * ps
    history = np.empty(max_iters)
    for i in range(max_iters):
        new_X = X + rng.uniform(-1, 1, ndim) * current_temperature
        new_X.clip(*bounds, out=new_X)  # 裁剪边界
        # 计算适应度的值
        new_cost = fitness_func(new_X, **kwargs)
        diff = new_cost - fitness_func(X, **kwargs)
        # 能量减少或者一定概率接受新解
        if diff < 0 or rng.random() < np.exp(-diff / current_temperature):
            np.copyto(X, new_X)  # 更新X
            if new_cost < best_fitness:
                np.copyto(best_solution, X)  # 更新最佳值
                best_fitness = new_cost
        history[i] = best_fitness
        current_temperature *= cooling_rate
    return best_solution, best_fitness, history[::ps]
