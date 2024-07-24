#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 下午9:18
# @Author  : ACHIEVE_DREAM
# @File    : sparrow_search_algorithm.py
# @Software: Pycharm

from typing import Callable, Any

import numpy as np

from ..chaos_mapping import tent


def sparrow_sa(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    population_size: int,
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int = 100,
    seed: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    麻雀搜索算法
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 边界
    :param max_iters: 最大迭代次数
    :param seed: 随机数种子
    :return:
    :Reference:
    https://blog.csdn.net/weixin_46838605/article/details/127290940
    """
    rng = np.random.default_rng(seed)
    population = rng.uniform(*bounds, size=(population_size, ndim))
    fitness = fitness_func(population, **kwargs)
    idx = fitness.argmin()
    best_solution = population[idx].copy()
    best_fitness = fitness[idx]
    ST = 0.6  # 预警值
    SD = 0.2  # 意识到有危险麻雀的比重
    producer_ratio = 0.2  # 发现者的比例
    producer_nums = int(population_size * producer_ratio)
    middle = population_size // 2
    history = np.empty(max_iters)
    for i in range(max_iters):
        idx = fitness.argmax()
        worst_solution = population[idx].copy()
        worst_fitness = fitness[idx].copy()
        # 发现者位置更新
        if rng.random() < ST:  # 搜索环境安全,广泛搜索
            population[:producer_nums] = population[:producer_nums] * np.exp(
                -i / (rng.random(size=(producer_nums, ndim)) * max_iters)
            )
        else:  # 发现捕食者, 回到安全区域
            population[:producer_nums] = population[
                :producer_nums
            ] + rng.standard_normal(size=(producer_nums, ndim))
        # population.clip(*bounds, out=population)
        population[:producer_nums].clip(*bounds, out=population[:producer_nums])
        fitness[:producer_nums] = fitness_func(population[:producer_nums], **kwargs)
        idx = fitness.argmin()
        current_best_solution = population[idx].copy()
        # 跟随者位置更新
        A = rng.choice([-1, 1], size=(middle - producer_nums, ndim))
        # 一半之前的跟随者
        population[producer_nums:middle] = (
            current_best_solution
            + np.abs(population[producer_nums:middle] - current_best_solution) * A
        )
        # 超过种群一半的跟随着 |producer|-----|follower|------|middle|------------|population|
        population[middle:] = rng.standard_normal(size=(middle, ndim)) * np.exp(
            (worst_solution - population[middle:])
            / np.repeat(np.arange(middle, population_size) ** 2, 2).reshape(-1, 2)
        )

        # 侦察到危险的麻雀进行位置更新
        b = rng.choice(population_size, size=int(population_size * SD), replace=False)
        idx = (fitness[b] > best_fitness).ravel()
        if idx.any():
            population[b][idx] = best_solution + rng.standard_normal(
                size=(idx.sum(), ndim)
            ) * np.abs(population[b][idx] - best_solution)
        idx = ~idx  # 没有提升的位置
        # idx = (fitness[b] == best_fitness).ravel()  # 没有提升的位置
        if idx.any():
            population[b][idx] += population[b][idx] + rng.uniform(
                -1, 1, size=(idx.sum(), ndim)
            ) * (
                np.abs(population[b][idx] - worst_solution)
                / (fitness[b][idx] - worst_fitness + 1e-5)[:, None]
            )
        population.clip(*bounds, out=population)
        fitness = fitness_func(population, **kwargs)
        idx = fitness.argmin()
        if fitness[idx] < best_fitness:
            best_solution[:] = population[idx].copy()
            best_fitness = fitness[idx]
        history[i] = best_fitness
    return best_solution, best_fitness, history


def sparrow_sa_tent(
    fitness_func: Callable[[np.ndarray, Any], np.ndarray],
    population_size: int,
    ndim: int,
    bounds: tuple[float, float],
    max_iters: int = 100,
    seed: int = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    麻雀搜索算法
    :param fitness_func: 适应度函数
    :param population_size: 种群大小
    :param ndim: 问题维度
    :param bounds: 边界
    :param max_iters: 最大迭代次数
    :param seed: 随机数种子
    :return:
    :Reference:
    https://blog.csdn.net/weixin_46838605/article/details/127290940
    """
    rng = np.random.default_rng(seed)
    population = tent(*bounds, size=(population_size, ndim))
    fitness = fitness_func(population, **kwargs)
    idx = fitness.argmin()
    best_solution = population[idx].copy()
    best_fitness = fitness[idx]
    ST = 0.6  # 预警值
    SD = 0.2  # 意识到有危险麻雀的比重
    producer_ratio = 0.2  # 发现者的比例
    producer_nums = int(population_size * producer_ratio)
    middle = population_size // 2
    history = np.empty(max_iters)
    for i in range(max_iters):
        idx = fitness.argmax()
        worst_solution = population[idx].copy()
        worst_fitness = fitness[idx].copy()
        # 发现者位置更新
        if rng.random() < ST:  # 搜索环境安全,广泛搜索
            population[:producer_nums] = population[:producer_nums] * np.exp(
                -i / (rng.random(size=(producer_nums, ndim)) * max_iters)
            )
        else:  # 发现捕食者, 回到安全区域
            population[:producer_nums] = population[
                :producer_nums
            ] + rng.standard_normal(size=(producer_nums, ndim))
        # population.clip(*bounds, out=population)
        population[:producer_nums].clip(*bounds, out=population[:producer_nums])
        fitness[:producer_nums] = fitness_func(population[:producer_nums], **kwargs)
        idx = fitness.argmin()
        current_best_solution = population[idx].copy()
        # 跟随者位置更新
        A = rng.choice([-1, 1], size=(middle - producer_nums, ndim))
        # 一半之前的跟随者
        population[producer_nums:middle] = (
            current_best_solution
            + np.abs(population[producer_nums:middle] - current_best_solution) * A
        )
        # 超过种群一半的跟随着 |producer|-----|follower|------|middle|------------|population|
        population[middle:] = rng.standard_normal(size=(middle, ndim)) * np.exp(
            (worst_solution - population[middle:])
            / np.repeat(np.arange(middle, population_size) ** 2, 2).reshape(-1, 2)
        )

        # 侦察到危险的麻雀进行位置更新
        b = rng.choice(population_size, size=int(population_size * SD), replace=False)
        idx = (fitness[b] > best_fitness).ravel()
        if idx.any():
            population[b][idx] = best_solution + rng.standard_normal(
                size=(idx.sum(), ndim)
            ) * np.abs(population[b][idx] - best_solution)
        idx = ~idx  # 没有提升的位置
        # idx = (fitness[b] == best_fitness).ravel()  # 没有提升的位置
        if idx.any():
            population[b][idx] += population[b][idx] + rng.uniform(
                -1, 1, size=(idx.sum(), ndim)
            ) * (
                np.abs(population[b][idx] - worst_solution)
                / (fitness[b][idx] - worst_fitness + np.spacing(0))[:, None]
            )
        population.clip(*bounds, out=population)
        fitness = fitness_func(population, **kwargs)
        idx = fitness.argmin()
        if fitness[idx] < best_fitness:
            best_solution[:] = population[idx].copy()
            best_fitness = fitness[idx]
        history[i] = best_fitness
    return best_solution, best_fitness, history
