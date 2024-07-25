#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 上午9:38
# @Author  : ACHIEVE_DREAM
# @File    : slap_swarm_algorithm.py
# @Software: Pycharm
# 樽海鞘算法
from typing import Callable, Any

import numpy as np


# noinspection DuplicatedCode
def ssa(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    population_size: int,
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int,
    seed: int = None,
    **kwargs,
):
    """
    樽海鞘群算法
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 问题边界
    :param max_iters: 迭代次数
    :param seed: 随机数种子
    :return: 最优解, 最优解对应的适应度, 迭代历史记录
    "Reference：
        https://blog.csdn.net/Logic_9527/article/details/136634421
        https://www.mathworks.com/matlabcentral/fileexchange/63745-ssa-salp-swarm-algorithm
    """
    rng = np.random.default_rng(seed)
    # 种群初始化
    P = rng.uniform(*bounds, size=(population_size, ndim))
    fitness = fitness_func(P, **kwargs)
    idx = fitness.argmin()
    # 食物的位置
    best_solution = P[idx].copy()
    best_fitness = fitness[idx]
    history_fitness = np.empty(max_iters)
    # 开始迭代
    for i in range(max_iters):
        c1 = 2 * np.exp(-((4 * (i + 1) / max_iters) ** 2))
        c3 = rng.choice([-1, 1], size=(population_size // 2, ndim))
        # leaders, 前一半种群
        # 领导者在食物附近移动
        P[: population_size // 2] = best_solution + c3 * c1 * rng.uniform(
            *bounds, size=(population_size // 2, ndim)
        )
        # followers, P(i) = (P(i - 1) + P(i)) / 2， 后一半种群
        P[population_size // 2 :] = 0.5 * (
            P[population_size // 2 - 1 : -1] + P[population_size // 2 :]
        )
        # 超出范围的解重置为边界值
        P.clip(*bounds, out=P)
        # 重新计算适应度
        fitness[:] = fitness_func(P, **kwargs)
        # 找出最佳适应度的位置
        idx = fitness.argmin()
        if fitness[idx] < best_fitness:
            # 更新食物位置
            best_solution = P[idx].copy()
            best_fitness = fitness[idx]
        history_fitness[i] = best_fitness
    return best_solution, best_fitness, history_fitness
