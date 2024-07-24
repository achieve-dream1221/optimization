#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 下午1:24
# @Author  : ACHIEVE_DREAM
# @File    : apo.py
# @Software: Pycharm
from typing import Any
from typing import Callable

import numpy as np


def apo(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    population_size: int,
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int = 100,
    neighbor: int = 1,
    pf_max: float = 0.2,
    seed: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    人工原生动物优化器(APO)
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 边界
    :param max_iters: 迭代次数
    :param neighbor: 邻居个数
    :param pf_max: 比例分数最大值
    :param seed: 随机数种子
    :return: 最佳解, 最佳值, 历史值
    Reference:
    [1] Wang, Xiaopeng, Václav Snášel, Seyedali Mirjalili, Jeng-Shyang Pan, Lingping Kong和Hisham A. Shehadeh.
    《Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization》.
     Knowledge-Based Systems 295 (2024年7月8日): 111737. https://doi.org/10.1016/j.knosys.2024.111737.
    [2] https://blog.csdn.net/yuchunyu12/article/details/137741833
    [3] https://www.mathworks.com/matlabcentral/fileexchange/162656-artificial-protozoa-optimizer
    """
    rng = np.random.default_rng(seed)
    ps = population_size
    eps = np.spacing(1)
    P = rng.uniform(*bounds, size=(ps, ndim))
    NP = np.empty_like(P)
    epn = np.zeros((neighbor, ndim))  # EPN 表示配对邻居的影响
    fitness = fitness_func(P, **kwargs)
    best_index = fitness.argmin()
    best_solution = P[best_index].copy()
    best_fitness = fitness[best_index]
    history = np.empty(max_iters)
    for it in range(max_iters):
        a = fitness.argsort().ravel()
        P[:] = P[a]  # 按照适应度进行排序
        fitness[:] = fitness[a]
        pf = pf_max * rng.random()  # proportion fraction
        # 随机选择不重复的个体位置
        ri = rng.choice(ps, size=np.ceil(ps * pf).astype(int), replace=False)
        # 休眠和繁殖的概率
        pdr = 0.5 * (1 + np.cos((1 - ri / ps)) * np.pi)
        rand = rng.random(size=ri.size)
        # 休眠
        index = ri[rand < pdr]
        # 产生新的个体
        NP[index] = rng.uniform(*bounds, size=(index.size, ndim))
        # 繁殖
        index = ~index
        if index.any():  # 确保有繁殖的个体
            mr = (rng.random((index.size, ndim)) > 0.5).astype(int)
            NP[index] = (
                P[index]
                + rng.uniform(-1, 1, size=(index.size, ndim))
                * rng.uniform(*bounds, size=(index.size, ndim))
                * mr
            )
        # 原生动物觅食：
        not_in_ri = np.setdiff1d(np.arange(ps), ri)  # 不属于ri的个体位置
        pah = 0.5 * (1 + np.cos(it / max_iters * np.pi))  # 自养和异养行为的概率。
        rand = rng.random(size=not_in_ri.size)
        index = not_in_ri[rand < pah]  # 在自养生物中觅食
        mf = (rng.random((not_in_ri.size, ndim)) > 0.5).astype(int)
        j = rng.choice(ps, size=index.size)  # j 表示第 j 个随机选择的原生动物
        # km: k-, kp:k+ 自养
        # Foraging factor： 觅食因子
        f = rng.random(size=not_in_ri.size) * (1 + np.cos(it / max_iters * np.pi))
        for i, idx in enumerate(index.data):
            for k in range(neighbor):
                if idx == 0:  # 第一个没有km
                    km, kp = idx, rng.choice(ps)
                elif idx == ps - 1:  # 最后一个没有kp
                    km, kp = rng.choice(ps - 1), idx
                else:
                    km, kp = rng.choice(idx), idx + rng.choice(ps - idx)
                wa = np.exp(-np.abs(fitness[km] / (fitness[kp] + eps)))  # wa :自养权重
                epn[k] = wa * (P[km] - P[kp]) + eps  # EPN 表示配对邻居的影响
            NP[idx] = P[idx] + f[i] * (P[j[i]] - P[idx] + 1 / epn.sum()) * mf[i]
        # 异养觅食
        index = ~index
        if index.any():  # 确保有异养的个体
            for i, idx in enumerate(index.data):
                for k in range(1, neighbor + 1):
                    if idx == 0:  # 第一个没有imk
                        imk, ipk = idx, k
                    elif idx == ps - 1:  # 最后一个没有ipk
                        imk, ipk = ps - 1 - k, idx
                    else:
                        imk, ipk = idx - k, idx + k
                    imk = max(imk, 1)
                    ipk = min(ipk, ps - 1)
                    # 异养权重因子
                    wh = np.exp(-np.abs(fitness[imk] / (fitness[ipk] + eps)))
                    epn[k - 1] = wh * (P[imk] - P[ipk]) + eps
                x_near = (1 + rng.uniform(-1, 1, size=ndim) * (1 - it / max_iters)) * P[
                    idx
                ]
                NP[idx] = P[idx] + f[i] * (x_near - P[idx] + 1 / epn.sum()) * mf[i]
        # 裁剪边界
        NP.clip(*bounds, out=NP)
        new_fitness = fitness_func(NP, **kwargs)
        improved = (new_fitness < fitness).ravel()
        P[improved] = NP[improved]
        fitness[improved] = new_fitness[improved]
        index = fitness.argmin()
        if fitness[index] < best_fitness:
            best_solution = P[index].copy()
            best_fitness = fitness[index]
        history[it] = best_fitness
    return best_solution, best_fitness, history
