#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 上午10:07
# @Author  : ACHIEVE_DREAM
# @File    : pso.py
# @Software: Pycharm
from typing import Callable, Any

import numpy as np


def pso(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    population_size: int,
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int = 100,
    v_bounds: tuple[float, float] = (-1, 1),
    c1: float = 0.5,
    c2: float = 0.3,
    w: float = 0.8,
    seed: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    粒子群算法
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 问题边界
    :param max_iters: 迭代次数
    :param v_bounds: 速度边界
    :param c1: 算法常数1
    :param c2: 常数2
    :param w: 惯性权重
    :param seed: 随机数种子
    :return: 问题解, 最优值, 历史值
    :Reference:
        https://blog.csdn.net/C1172440795/article/details/125837484
        https://blog.csdn.net/weixin_52026057/article/details/129247540
    """
    rng = np.random.default_rng(seed)
    # 初始化粒子群
    P = rng.uniform(*bounds, size=(population_size, ndim))
    V = rng.uniform(*v_bounds, size=(population_size, ndim))
    F = fitness_func(P, **kwargs)
    # 每个个体的历史最佳位置
    P_B = P.copy()
    idx = F.argmin()
    # 全局最优解
    best_solution = P[idx].copy()
    # 全局最优适应度值
    best_fitness = F[idx].copy()
    # 记录迭代的适应度变化
    history = np.empty(max_iters)
    for i in range(max_iters):
        # 更新速度和位置
        r1 = rng.random(size=(population_size, ndim))
        r2 = rng.random(size=(population_size, ndim))
        V[:] = w * V + c1 * r1 * (P_B - P) + c2 * r2 * (best_solution - P)
        V[:] = V.clip(*v_bounds)
        P[:] += V
        P[:] = P.clip(*bounds)
        # 更新历史最优位置和全局最优位置
        F_new = fitness_func(P, **kwargs)
        # 更新单个个体最优
        improved = (F_new < F).ravel()
        P_B[improved] = P[improved]
        F[improved] = F_new[improved]
        # 更新全局最优
        idx = F.argmin()
        if F[idx] < best_fitness:
            best_solution[:] = P[idx].copy()
            best_fitness = F[idx].copy()
        history[i] = best_fitness.copy()
    return best_solution, best_fitness, history
