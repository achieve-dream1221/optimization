#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 上午10:10
# @Author  : ACHIEVE_DREAM
# @File    : harris_hawk_optimization.py
# @Software: Pycharm
from typing import Callable, Any

import numpy as np


def hho(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    population_size: int,
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int = 100,
    seed: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    哈里斯鹰优化算法
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 边界
    :param max_iters: 最大迭代次数
    :param seed: 随机数种子
    :return: 最佳解, 最佳值, 历史值
    https://blog.csdn.net/m0_68737136/article/details/132334954
    """
    rng = np.random.default_rng(seed)
    P = rng.uniform(*bounds, size=(population_size, ndim))
    fitness = fitness_func(P, **kwargs)
    idx = fitness.argmin()
    best_fitness = fitness[idx].copy()
    best_solution = P[idx].copy()
    history = np.empty(max_iters)
    for i in range(max_iters):
        # 逃逸能量
        Esc_E = 2 * (1 - i / max_iters) * rng.uniform(*[-1, 1], size=population_size)
        # 随机数小于0.5
        q = rng.uniform(size=population_size) < 0.5
        # 大于等于0.5
        nq = ~q
        # 绝对值大于1的情况
        idx = np.abs(Esc_E) >= 1
        # 进行与运算: 即能量绝对值大于等于1并且随机数小于0.5的个体下标
        idx1 = idx & q
        rand = rng.choice(P, size=idx1.sum())  # 随机选择的个体, (idx1.sum(), ndim)
        P[idx1] = rand - rng.random(size=(idx1.sum(), ndim)) * np.abs(
            rand - 2 * rng.random(size=(idx1.sum(), ndim)) * P[idx1]
        )
        # 进行与运算: 即能量绝对值大于等于1并且随机数大于等于0.5的个体下标
        idx1 = idx & nq
        P[idx1] = (
            best_solution
            - P.mean(axis=0)
            - rng.random(size=(idx1.sum(), ndim))
            * rng.uniform(*bounds, size=(idx1.sum(), ndim))
        )
        # 绝对值小于0.5的两种情况
        idx[:] = np.abs(Esc_E) < 0.5
        # 进行与运算: 即能量绝对值小于0.5并且随机数小于0.5的个体下标
        idx1 = idx & q
        P[idx1] = best_solution - Esc_E[idx1, None] * np.abs(best_solution - P[idx1])
        # 进行与运算: 即能量绝对值小于0.5并且随机数大于等于0.5的个体下标
        idx1 = idx & nq
        jump_strength = rng.uniform(0, 2, size=idx1.sum())
        P1 = best_solution - Esc_E[idx1, None] * np.abs(
            jump_strength[:, None] * best_solution - P.mean(axis=0)
        )
        f = fitness_func(P[idx1], **kwargs)
        a = (fitness_func(P1, **kwargs) < f).ravel()
        P[np.where(idx1)[0][a]] = P1[a].copy()
        a = ~a  # 没有提升的坐标
        if a.any():
            P2 = (
                best_solution
                - Esc_E[idx1][a, None]
                * np.abs(jump_strength[a][:, None] * best_solution - P.mean(axis=0))
                + rng.random(size=(a.sum(), ndim))
            )  # TODO: LEVY
            b = (fitness_func(P2, **kwargs) < f[a]).ravel()
            P[np.where(idx1)[0][a][b]] = P2[b].copy()
        # 绝对值大于等于0.5, 小于1的两种情况
        idx[:] = (np.abs(Esc_E) >= 0.5) & (np.abs(Esc_E) < 1)
        # 进行与运算: 即能量绝对值在0.5和1之间并且随机数小于0.5的个体下标
        idx1 = idx & q
        jump_strength = rng.uniform(0, 2, size=idx1.sum())
        P[idx1] = (best_solution - P[idx1]) - Esc_E[idx1, None] * np.abs(
            jump_strength[:, None] * best_solution - P[idx1]
        )
        # 进行与运算: 即能量绝对值在0.5和1之间并且随机数大于等于0.5的个体下标
        idx1 = idx & nq
        jump_strength = rng.uniform(0, 2, size=idx1.sum())
        P1 = best_solution - Esc_E[idx1, None] * np.abs(
            jump_strength[:, None] * best_solution - P[idx1]
        )
        f = fitness_func(P[idx1], **kwargs)
        a = (fitness_func(P1, **kwargs) < f).ravel()
        P[np.where(idx1)[0][a]] = P1[a].copy()
        a = ~a
        if a.any():
            P2 = (
                best_solution
                - Esc_E[idx1][a, None]
                * np.abs(jump_strength[a][:, None] * best_solution - P[idx1][a])
                + rng.random(size=(a.sum(), ndim))
            )  # TODO: LEVY
            b = (fitness_func(P2, **kwargs) < f[a]).ravel()
            P[np.where(idx1)[0][a][b]] = P2[b].copy()
        P.clip(*bounds, out=P)
        fitness[:] = fitness_func(P, **kwargs)
        idx = fitness.argmin()
        if fitness[idx] < best_fitness:
            best_fitness = fitness[idx].copy()
            best_solution = P[idx].copy()
        history[i] = best_fitness
    return best_solution, best_fitness, history
