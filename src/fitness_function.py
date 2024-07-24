#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 上午9:41
# @Author  : ACHIEVE_DREAM
# @File    : fitness_function.py
# @Software: Pycharm
# 适应度函数合集
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Fitness:
    func: Callable
    bounds: tuple
    solution: tuple
    min: float


def get_fitness_func(func_name: str) -> Fitness:
    """
    获取适应度函数
    :param func_name: 适应度函数名称， uf*: 单峰函数， mf*, 多峰函数， f*: CEC2022基准测试函数
    :return:
    """
    func_dict = {
        "uf1": Fitness(UniModal.f1, (-100, 100), (0,), 0),
        "uf2": Fitness(UniModal.f2, (-10, 10), (0,), 0),
        "mf1": Fitness(MultiModal.f1, (-5.12, 5.12), (0,), 0),
        "mf2": Fitness(MultiModal.f2, (-5, 5), (0,), -1.0316),
        "f1": Fitness(CEC2022.f1, (-100, 100), (0,), 0),
        "f2": Fitness(CEC2022.f2, (-5.12, 5.12), (0,), 0),
        "f3": Fitness(CEC2022.f3, (-100, 100), (0,), 0),
        "f4": Fitness(CEC2022.f4, (-100, 100), (0,), 0.25),
        "f5": Fitness(CEC2022.f5, (-100, 100), (1,), 0),
        "f6": Fitness(CEC2022.f6, (-100, 100), (0,), 0),
        "f7": Fitness(CEC2022.f7, (-100, 100), (0,), 0),
    }
    assert func_name in func_dict.keys(), f"{func_name} is not in func_dict"
    return func_dict[func_name]


class UniModal:
    """单峰基准测试"""

    @staticmethod
    def f1(X: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        f(x) = \sum_i x_i^2
        bound: [-100, 100]
        min: 0
        :param X: 矩阵(个体， 维度)
        :return: 每一个个体的适应度
        """
        assert X.ndim <= 2, "X.ndim must be 1 or 2"
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X**2).sum(axis=1)

    @staticmethod
    def f2(X: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        f(x) = \sum_i |x_i| + \prod |x_i|
        bound: [-10, 10]
        min: 0
        :param X: 矩阵(个体， 维度)
        :return: 每一个个体的适应度
        """
        assert X.ndim <= 2, "X.ndim must be 1 or 2"
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.abs(X).sum(axis=1) + np.abs(X.prod(axis=1))


class MultiModal:
    """多峰基准测试"""

    @staticmethod
    def f1(X: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        f(x) = \sum_i x_i^2 - 10 * cos(2 * pi * x_i) + 10
        bound: [-5.12, 5.12]
        min: 0
        :param X: 矩阵(个体， 维度)
        :return: 每一个个体的适应度
        """
        assert X.ndim <= 2, "X.ndim must be 1 or 2"
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X**2 - 10 * np.cos(X * 2 * np.pi) + 10).sum(axis=1)

    @staticmethod
    def f2(X: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        f(x) = 4 * x_1^2 - 2.1 * x_1^4 + x_1^6 / 3 + x_1 * x_2 - 4 * x_2^2 + 4 * x_2^4
        bound: [-5, 5]
        min: -1.0316
        :param X: (rows, 2) matrix
        :return: fitness
        """
        assert X.ndim == 2 and X.shape[1] == 2, "X.ndim must be 2 and X.shape[1] == 2"
        x1 = X[:, 0]
        x2 = X[:, 1]
        return 4 * x1**2 - 2.1 * x1**4 + x1**6 / 3 + x1 * x2 - 4 * x2**2 + 4 * x2**4


class CEC2022:
    """
    Reference: https://www.kaggle.com/code/kooaslansefat/cec-2022-benchmark/notebook
    """

    @staticmethod
    def f1(X: np.ndarray, **kwargs) -> np.ndarray:
        """
        F1: Bent Cigar Function F1：弯曲雪茄函数
        bounds: [-100, 100]
        min: f(0) = 0
        :param X: n 维向量
        :return: result
        """
        # assert X.ndim >= 2, "X.ndim must >= 2"
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # axis=1: 对每一行求和
        return X[:, 0] ** 2 + 1e6 * (X[:, 1:] ** 2).sum(axis=1)

    @staticmethod
    def f2(X: np.ndarray, **kwargs) -> np.ndarray:
        """
        F2: Rastrigin’s Function F2：拉斯特里金函数
        bounds: [-5.12， 5.12]
        min: f(0) = 0
        :param X: n维向量
        :return: result
        """
        # assert X.ndim >= 2, "X.ndim must be 2"
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X**2 - 10 * np.cos(2 * np.pi * X) + 10).sum(axis=1)

    @staticmethod
    def f3(X: np.ndarray, **kwargs) -> np.ndarray:
        """
        High Conditioned Elliptic Function
        F3：高条件椭圆函数
        bounds = [-100, 100]
        min: f(0,0) = 0
        :param X: n维向量
        :return: result
        """
        assert X.ndim >= 2, "X.ndim must be 2"
        a = 1e6
        D = X.shape[1]
        return (a ** (np.arange(D) / (D - 1)) * X**2).sum(axis=1)

    @staticmethod
    def f4(X: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        HGBat Function
        F4：HGBat 函数
        min: f(0,...) = 0.25
        :latex:
           f(\textbf{x}) = \left| \left( \sum_{i=1}^D x_i^2 \right)^2 - \left( \sum_{i=1}^D x_i \right)^2
            \right|^{\frac{1}{2}} + \frac{0.5 \sum_{i=1}^D x_i^2 + \sum_{i=1}^D x_i}{D} + 0.5
        :param X: n维向量
        :return: result
        """
        D = X.shape[1]
        return (
            (np.abs((X**2).sum(axis=1) - X.sum(axis=1) ** 2)) ** 0.5
            + (0.5 * (X**2).sum(axis=1) + X.sum(axis=1)) / D
            + 0.5
        )
        # a = np.asarray([-10, -5])
        # return ((X - a) ** 2).sum(axis=1) + (X - a).sum(axis=1) ** 2 / 1e6

    @staticmethod
    def f5(X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Rosenbrock函数, 具有几个局部最小值
        min: f(1,1) = 0
        :param X: n维向量
        :return: result
        """
        assert X.ndim == 2, "X.ndim must be 2"
        D = X.shape[1]
        i = np.arange(D - 1)
        return 100 * (X[:, i] ** 2 - X[:, i + 1]) ** 2 + (1 - X[:, i + 1]) ** 2

    @staticmethod
    def f6(X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Ackley’s Function F7：阿克利函数, 具有几个局部最小值
        min: f(0,0) = 0
        :param X: n维向量
        :return: result
        """
        D = X.shape[1]
        return (
            -20 * np.exp(-0.2 * np.sqrt((X**2).sum(axis=1) / D))
            - np.exp(np.cos(2 * np.pi * X).sum(axis=1) / D)
            + 20
            + np.e
        )

    @staticmethod
    def f7(X: np.ndarray, **kwargs):
        """
        Expanded Schaffer’s Function
        min: f(0,0)=0
        𝑓3(𝐱) = g(𝑥1, 𝑥2) + g(𝑥2, 𝑥3) + ... + g(𝑥𝐷−1, 𝑥𝐷) + g(𝑥𝐷, 𝑥1)
        :param X: n维向量
        :return: result
        """

        def schaffer(x, y):
            return (
                0.5
                + (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5)
                / (1 + 0.001 * (x**2 + y**2)) ** 2
            )

        # 错位相加
        return np.sum(schaffer(X[:, :-1], X[:, 1:]), axis=1)
