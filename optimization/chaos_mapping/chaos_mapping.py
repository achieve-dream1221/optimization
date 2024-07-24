#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 上午9:37
# @Author  : ACHIEVE_DREAM
# @File    : chaos_mapping.py
# @Software: Pycharm
import numpy as np


def tent(
    low: float, high: float, size: tuple | int, z: float = 0.5, u: float = 1.90
) -> np.ndarray:
    """
    tent map
    References:
    [1]: https://en.wikipedia.org/wiki/Tent_map
    :param low: 下界
    :param high: 上界
    :param size: 生成的大小
    :param z: 初始值
    :param u: 参数
    :return: 生成数据
    """
    assert 0 < z < 1, "z must be in (0, 1)"
    assert 1.8 < u < 2, "u must be in (0, 2)"
    length = np.prod(size)
    data = np.empty(length)
    for i in range(length):
        z = u * min(z, 1 - z)
        data[i] = z
    return (low + data * (high - low)).reshape(size)


def cat(
    low: float, high: float, size: tuple | int, z: float = 1, u: float = 1
) -> np.ndarray:
    r"""
    cat map
    :param low: 下界
    :param high: 上界
    :param size: 生成的大小
    :param z: 初始值
    :param u: 参数
    :return: 生成数据
    """
    x, y = 0.1, 0.1
    length = np.prod(size)
    data = np.empty(length)
    for i in range(length):
        x_new = (x + z * y) % 1.0
        y_new = (u * x + (1 + u * z) * y) % 1.0
        x, y = x_new, y_new
        data[i] = x_new
    return (low + data * (high - low)).reshape(size)


def logistic(
    low: float, high: float, size: tuple | int, z: float = 0.5201314, u: float = 3.99
) -> np.ndarray:
    """
    logistic map: logistic混沌映射
    :math: z(k+1) = u * z(k) * (1 - z(k)), 推荐u取[3.83, 4],这样才会有明显的混沌性
    References:
    [1]: https://en.wikipedia.org/wiki/Logistic_map#Behavior_dependent_on_r
    :param low: 下界
    :param high: 上界
    :param size: 大小
    :param z: 初始值
    :param u: r
    :return: 大小为shape的列表
    """
    assert 3.83 <= u < 4, "u must be in [3.83, 4)"
    # z(k+1) = u * z(k) * (1 - z(k))
    length = np.prod(size)
    data = np.empty(length)
    for i in range(length):
        z = u * z * (1 - z)
        data[i] = z
    return (low + data * (high - low)).reshape(size)


def sine(
    low: float, high: float, size: tuple | int, z: float = 0.1, u: float = 0.99
) -> np.ndarray:
    r"""
    sine map
    :math: x_{i+1} = \mu sin(\pi x_i)
    :param low: 下界
    :param high: 上界
    :param size: 生成的大小
    :param z: 初始值
    :param u: 参数
    :return: 生成数据
    """
    assert 0 < z < 1, "z must be in (0, 1)"
    assert (0.87 < u < 0.93) or (
        0.95 < u < 1
    ), "mu must be in (0.87, 0.93) or (0.95, 1)"
    length = np.prod(size)
    data = np.empty(length)
    for i in range(length):
        z = u * np.sin(np.pi * z)
        data[i] = z
    return (low + data * (high - low)).reshape(size)
