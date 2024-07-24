# 关于

不同优化算法的python实现

author: wulei7217@gmail.com

# 如何使用

```shell
# 安装依赖
pip install -r requirements.lock
# 运行测试脚本
## 混沌映射测试
python src/bench_chaos_mapping.py
## 优化算法测试
python src/bench_optimization.py
```

# 目录说明

| 目录           | 说明                   |
|--------------|----------------------|
| config       | 配置类(不需要更改)           |
| optimization | 原实现在src下, 其他目录均为改进策略 |
| src          | 算法性能测试               |

# 已实现的优化算法

1. [APO (Artificial Protozoa Optimizer)](optimization/src/artificial_protozoa_optimization.py)
2. [HHO (harris hawk optimization)](optimization/src/harris_hawk_optimization.py)
3. [PSO (particle swarm optimization)](optimization/src/particle_swarm_optimization.py)
4. [SSA (salp swarm algorithm)](optimization/src/salp_swarm_algorithm.py)
5. [SA (simulated annealing)](optimization/src/simulated_annealing.py)
6. [SO (snake optimization)](optimization/src/snake_optimizer.py)
7. [SSA (sparrow search algorithm)](optimization/src/sparrow_search_algorithm.py)
