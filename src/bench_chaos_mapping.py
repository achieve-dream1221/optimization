from typing import Callable
import time

import numpy as np
import matplotlib.pyplot as plt
from optimization.chaos_mapping import *


def bench(funcs: list[Callable]):
    funcs_size = len(funcs)
    size = 1000
    bounds = [0, 1]
    datas = np.empty((funcs_size, size))
    used_times = np.empty(funcs_size)

    for i, func in enumerate(funcs):
        start = time.perf_counter()
        datas[i] = func(*bounds, size=size)
        used_times[i] = time.perf_counter() - start
    names = [f.__name__ for f in funcs]
    names[-1] = "random"
    plt.figure(figsize=(8, 4))
    for i, data in enumerate(datas):
        data: np.ndarray
        plt.subplot(1, funcs_size, i + 1)
        plt.hist(data)
        # plt.title(f"{names[i]} 分布直方图")
        plt.title(f"{names[i]}")
        print(f"{names[i]}的平均值: {data.mean()}\t耗时: {used_times[i]} s")
    plt.tight_layout()
    plt.savefig(f"data/不同混沌映射函数分布直方图对比图.svg")

    # plt.figure(figsize=(8, 6))
    # for i, data in enumerate(datas):
    #     data: np.ndarray
    #     plt.subplot(funcs_size, 2, 2 * i + 1)
    #     # plt.plot(sorted(data), [i / size for i in range(1, size + 1)])
    #     plt.plot(sorted(data), np.linspace(0, 1, size))
    #     plt.title(f"{names[i]} CDF")
    #     plt.subplot(funcs_size, 2, 2 * i + 2)
    #     plt.hist(data)
    #     plt.title(f"{names[i]} 分布直方图")
    #     print(f"{names[i]}的平均值: {data.mean()}\t耗时: {used_times[i]} s")
    #
    # plt.tight_layout()
    # plt.savefig("data/不同随机函数CDF和直方图对比图.svg")

    # for i, data in enumerate(datas):
    #     plt.figure()
    #     plt.plot(data, ".", fillstyle="none")
    #     plt.title(f"{names[i]} 分布图")
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(18398203582)
    # cauchy = lambda low, high, size: low + rng.standard_cauchy(size) * (high - low)
    # bench([tent, np.random.uniform, cauchy])
    bench([logistic, tent, rng.uniform])
