#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 下午3:37
# @Author  : ACHIEVE_DREAM
# @File    : snake.py
# @Software: Pycharm
from typing import Callable, Any

import numpy as np


# noinspection DuplicatedCode
def so(
        fitness_func: Callable[[np.ndarray, Any], np.ndarray],
        population_size: int,
        ndim: int,
        bounds: tuple[float, float],
        max_iters: int = 100,
        seed: int = None,
        **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 边界
    :param max_iters: 迭代次数
    :param seed: 随机数种子
    :return: 最优解, 最佳值, 历史值
    :Reference:
        https://blog.csdn.net/weixin_43821559/article/details/125041838
    """
    # 初始化阶段
    # 有没有食物的阈值
    food_threshold = 0.25
    # 温度适不适合交配的阈值
    temp_threshold = 0.6
    # 模式阈值，在下面会使用到，当产生的随机值小于模式阈值就进入战斗模式，否则就进入交配模式
    mode_threshold = 0.6
    # 常量c1,下面计算食物的质量的时候会用到
    c1 = 0.5
    # 常量c2,下面更新位置的时候会用到
    c2 = 0.05
    # 常量c3,用于战斗和交配
    c3 = 2
    # 一个非常接近0的数, 防止除0操作
    eps = np.spacing(0)
    rng = np.random.default_rng(seed)
    # P: 原始种群
    P = rng.uniform(*bounds, size=(population_size, ndim))
    NP = P.copy()
    fitness = fitness_func(P, **kwargs)
    idx = fitness.argmin()
    best_solution = P[idx].copy()
    best_fitness = fitness[idx]
    # 初始化分离种群: 雄性, 雌性
    fs, ms = population_size // 2, population_size - population_size // 2
    FP, MP = NP[:fs], NP[fs:]
    FF, MF = fitness[:fs], fitness[fs:]

    # 迭代阶段
    history = np.empty(max_iters)
    for i in range(max_iters):
        male_best_idx = MF.argmin()
        female_best_idx = FF.argmin()
        t = np.exp(-i / max_iters)
        food_quality = c1 * np.exp((t - max_iters) / max_iters)
        # 先判断食物的质量是不是超过了阈值, 没有, 就寻找食物
        if food_quality < food_threshold:
            random_indexes = rng.integers(0, fs, size=fs)
            Af = np.exp(-FF[random_indexes] / (FF + eps))
            FP[:] = P[:fs][random_indexes] + rng.choice(
                [-1, 1], size=(fs, ndim)
            ) * c2 * Af[:, None] * rng.uniform(*bounds, size=(fs, ndim))
            random_indexes = rng.integers(0, ms, size=ms)
            Am = np.exp(-MF[random_indexes] / (MF + eps))
            MP[:] = P[fs:][random_indexes] + rng.choice(
                [-1, 1], size=(ms, ndim)
            ) * c2 * Am[:, None] * rng.uniform(*bounds, size=(ms, ndim))
        elif t > temp_threshold:  # 进入探索阶段, 表示当前是热的
            # 热了就不进行交配，开始向食物的位置进行移动
            FP[:] = best_solution + rng.choice(
                [-1, 1], (fs, ndim)
            ) * c3 * t * rng.random(size=(fs, ndim)) * (best_solution - P[:fs])
            MP[:] = best_solution + rng.choice(
                [-1, 1], (ms, ndim)
            ) * c3 * t * rng.random(size=(ms, ndim)) * (best_solution - P[fs:])
        else:
            # 如果当前的温度是比较的冷的，就比较适合战斗和交配,生成一个随机值来决定是要战斗还是要交配
            if rng.random() > mode_threshold:
                # 战斗模式
                fight = np.exp(-MF[male_best_idx] / (FF + eps))
                FP[:] = P[:fs] + c3 * fight * rng.random((fs, ndim)) * (
                        food_quality * FP[female_best_idx] - P[:fs]
                )
                fight = np.exp(-FF[female_best_idx] / (MF + eps))
                MP[:] = P[fs:] + c3 * fight * rng.random((ms, ndim)) * (
                        food_quality * MP[male_best_idx] - P[fs:]
                )
            else:
                # 交配模式
                mate = np.exp(-MF / (FF + eps))
                FP[:] = P[:fs] + c3 * mate * rng.random((fs, ndim)) * (
                        food_quality * MP - P[:fs]
                )
                mate = np.exp(-FF / (MF + eps))
                MP[:] = P[:fs] + c3 * mate * rng.random((ms, ndim)) * (
                        food_quality * FP - P[:fs]
                )
            # 产蛋
            if rng.random() <= 0.5:
                # 更新最差适应度个体的位置
                FP[FF.argmax()] = rng.uniform(*bounds, size=ndim)
                MP[MF.argmax()] = rng.uniform(*bounds, size=ndim)
        # 将更新后的位置进行处理
        NP.clip(*bounds, out=NP)
        new_fitness = fitness_func(NP, **kwargs)
        improved = (new_fitness < fitness).ravel()
        # 更新
        fitness[improved] = new_fitness[improved]
        P[improved] = NP[improved]  # 只更新有提升的
        idx = fitness.argmin()
        if fitness[idx] < best_fitness:
            best_solution = P[idx].copy()
            best_fitness = fitness[idx].copy()
        history[i] = best_fitness
    return best_solution, best_fitness, history
