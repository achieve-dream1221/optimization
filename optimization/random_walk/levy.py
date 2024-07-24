#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/27 下午4:45
# @Author  : ACHIEVE_DREAM
# @File    : levy.py
# @Software: Pycharm
from typing import Iterator

import numpy as np


def levy_generator(
    beta: float = 1.5,
    use_gamma: bool = False,
    sigma: float = 0.6965745025576968,
    seed: int = None,
) -> Iterator[np.float64]:
    r"""
    levy飞行 Levy 分布往往比较复杂，所以通常使用 Mantegna 算法模拟
    ···latex
        S=\frac{\mu}{v^{\frac 1 \beta}}
        u服从N(0,\sigma ^2), v服从N(0,1)分布
        复杂形式:\sigma = \left \{ \frac {\Gamma(1+\beta)sin(\frac {\pi \beta} 2)}
        {\beta \Gamma(\frac{1+\beta}{2})2^\frac{\beta - 1}{2}} \right \}^{\frac 1 \beta}
    ···
    :param beta: 公式参数，通常为1.5
    :param use_gamma: 使用gamma函数求解, 此时sigma参数失效,因为sigma可通过gamma和beta计算得出
    :param sigma: 公式参数, 默认值为beta=1.5的计算结果，以提高运算速度
    :param seed: 随机数种子
    :return: 随机行走路径
    参考链接:
    https://blog.csdn.net/qq_43445553/article/details/128186445
    https://www.mathworks.com/matlabcentral/fileexchange/162446-levy-flight-distribution?s_tid=srchtitle_support_results_1_levy%20flight
    """
    assert 1 <= beta <= 3, "beta must be in [1, 3]"
    rng = np.random.default_rng(seed)
    if use_gamma:
        from scipy.special import gamma

        sigma = (
            (gamma(1 + beta) * np.sin(np.pi * beta / 2))
            / (beta * gamma((1 + beta) / 2) * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)
    while True:
        yield rng.normal(0, sigma) / (np.abs(rng.standard_normal()) ** (1 / beta))


#
# def levy_flight(
#     size: Union[int, Tuple[int, ...]],
#     beta: float = 1.5,
#     use_gamma: bool = True,
#     sigma: float = 1,
#     seed: int = None,
# ) -> np.ndarray:
#     r"""
#     levy飞行 Levy 分 布往往比较复杂，所以通常使用 Mantegna 算法模拟
#     ···latex
#         S=\frac{\mu}{v^{\frac 1 \beta}}
#         u服从N(0,\sigma ^2), v服从N(0,1)分布
#         复杂形式:\sigma = \left \{ \frac {\Gamma(1+\beta)sin(\frac {\pi \beta} 2)}
#         {\beta \Gamma(\frac{1+\beta}{2})2^\frac{\beta - 1}{2}} \right \}^{\frac 1 \beta}
#     ···
#     :param use_gamma: 使用gamma函数求解, 此时sigma参数失效,因为sigma可通过gamma和beta计算得出
#     :param size: 模拟步数
#     :param sigma: 公式参数, 标准差
#     :param beta: 公式参数，通常为1.5
#     :param seed: 随机数种子
#     :return: 随机行走路径
#     参考链接:
#     https://blog.csdn.net/qq_43445553/article/details/128186445
#     https://www.mathworks.com/matlabcentral/fileexchange/162446-levy-flight-distribution?s_tid=srchtitle_support_results_1_levy%20flight
#     """
#     assert 1 <= beta <= 3, "beta must be in [1, 3]"
#     rng = np.random.default_rng(seed)
#     if use_gamma:
#         from scipy.special import gamma
#
#         sigma = (
#             (gamma(1 + beta) * np.sin(np.pi * beta / 2))
#             / (beta * gamma((1 + beta) / 2) * (2 ** ((beta - 1) / 2)))
#         ) ** (1 / beta)
#     u = rng.normal(0, sigma, size=size)
#     v = rng.standard_normal(size)  # standard Normal distribution (mean=0, stdev=1).
#     return u / (np.abs(v) ** (1 / beta))
