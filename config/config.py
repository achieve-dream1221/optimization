#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 下午9:39
# @Author  : ACHIEVE_DREAM
# @File    : config.py
# @Software: Pycharm
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo


def load_config(config_path: str) -> dict:
    # 加载默认配置
    p = Path(config_path)
    assert p.exists(), f"{config_path} 不存在"
    config = tomllib.loads(p.read_text(encoding="utf8"))
    p = Path(config["save"]["root"])
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return config


# noinspection PyNestedDecorators
class SaveConfig(BaseModel):
    root: str = "data"
    history: str = "data/history.csv"
    result: str = "data/result.json"
    figure: str = "data/result.svg"

    @field_validator(
        "history",
        "result",
        "figure",
    )
    @classmethod
    def fix_path(cls, v: str, info: ValidationInfo) -> str:
        return f"{info.data['root']}/{v}"


# noinspection PyNestedDecorators
class DataConfig(BaseModel):
    population_size: int = 30
    ndim: int = 2
    bounds: tuple[float, float] = (-10, 10)
    max_iters: int = 100

    @field_validator("bounds")
    @classmethod
    def check_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"bound: {v}, bound[0] must be less than bound[1]")
        return v


# noinspection PyNestedDecorators
class Config(BaseModel):
    save: SaveConfig = Field(default_factory=SaveConfig)
    config: DataConfig = Field(default_factory=DataConfig)

    def __init__(self, config_path: str = None):
        """
        项目配置
        :param config_path: 配置文件的路径
        """
        if config_path is not None:
            super().__init__(**load_config(config_path))
        else:
            super().__init__()


default_settings = Config()
