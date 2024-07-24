#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 下午9:34
# @Author  : ACHIEVE_DREAM
# @File    : optimization_bench.py
# @Software: Pycharm
import csv
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import Config
from fitness_function import get_fitness_func
from optimization import *


def benchmark(funcs: list[Callable]):
    # 配置
    # settings = default_settings
    settings = Config("config.toml")
    ndim = settings.config.ndim
    funcs_size = len(funcs)
    # 适应度函数, 默认CEC2022的F1, uf*: 单峰函数(uf1, uf2...)， mf*, 多峰函数， f*: CEC2022基准测试函数
    f = get_fitness_func("uf1")
    settings.config.bounds = f.bounds
    print(settings)
    # 记录数据
    used_times = np.empty(funcs_size)  # 运行时间
    solutions = np.empty((funcs_size, ndim))
    fitness = np.empty(funcs_size)
    history = np.empty((funcs_size, settings.config.max_iters))
    markers = ["o", "s", "^", "*", "x", "d", "p", "h", "v", "D"]
    # 对每一个优化算法进行记录
    for i, func in enumerate(funcs):
        start = time.perf_counter()  # 精度的计时器
        best_solution, best_fitness, history_fitness = func(
            fitness_func=f.func, **settings.config.model_dump()
        )
        used_times[i] = time.perf_counter() - start
        history[i] = history_fitness
        solutions[i] = best_solution
        fitness[i] = best_fitness.item()
        plt.plot(history_fitness, marker=markers[i], fillstyle="none")
    # 写入数据
    # 保存数据配置
    writer = csv.writer(
        Path(settings.save.history).open("w", encoding="utf8"),
        lineterminator="\n",
    )
    names = [func.__name__ for func in funcs]
    writer.writerow(names)
    writer.writerows(history.T)
    df = pl.DataFrame(
        [
            pl.Series("优化算法", names, pl.Categorical),
            pl.Series("耗时/s", used_times, pl.Float32),
            pl.Series("算法最优解", solutions, pl.Array(pl.Float32, shape=ndim)),
            pl.Series(
                "理论最优解",
                [np.array([f.solution] * ndim)] * funcs_size,
                pl.Array(pl.Float32, shape=ndim),
            ),
            pl.Series("算法最优值", fitness, pl.Float32),
            pl.Series("理论最优值", [f.min] * funcs_size, pl.Float32),
        ]
    )
    print(df)
    df.write_json(settings.save.result)
    plt.legend(names)
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.savefig(settings.save.figure)
    plt.show()


if __name__ == "__main__":
    # 测试不同的优化算法
    benchmark([ssa, pso, so, sparrow_sa, ssa])
