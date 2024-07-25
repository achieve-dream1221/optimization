import numpy as np
import matplotlib.pyplot as plt
import time


# noinspection DuplicatedCode,PyUnresolvedReferences
def leach() -> tuple[np.ndarray, np.ndarray]:
    seed = None
    rng = np.random.default_rng(seed)
    node_nums = 200  # 节点数量
    mat_iters = 2000  # 迭代次数
    bound = 100  # 边界
    base_station = (50, 125)  # 基站
    p = 0.08  # 簇头概率,未计算最佳簇头数量
    # 传输模型参数
    E_consume = 5e-8  # 单位bit数据发送、接收所消耗的能量
    Efs = 1e-11  # 自由空间传输
    Emp = 1.3e-15  # 多径衰落信道
    ED = 5e-9  # 能量消耗常量（Energy Dissipation）,传输数据时所消耗的能量
    d0 = 87  # 信道切换阈值 ？
    packet_length = 4000  # 数据包大小
    control_packet_length = 100  # 控制数据包大小
    E0 = 0.5
    # E_min = 0.001  # 节点存活所需的最小能量
    # radius = 15  # 初始通信距离

    # 初始化, 节点随机分布
    nodes_is_normal = np.ones(
        node_nums, dtype=np.bool
    )  # type True: 普通节点, False: 当前节点为簇头节点
    nodes_is_selected = np.zeros(
        node_nums, dtype=np.bool
    )  # False: 没有当选过簇头, True: 当选过
    nodes_x_y = np.empty((node_nums, 2))
    nodes_x_y[:, [0, 1]] = rng.uniform(0, bound, (node_nums, 2))  # xd yd x, y坐标
    nodes_cluster_p = rng.random(node_nums)  # temp_rand 随机数, 选为簇头的概率
    nodes_power = np.ones(node_nums) * E0  # power 初始能量
    nodes_cluster = np.zeros(
        node_nums, dtype=np.int32
    )  # CH 保存普通节点的簇头节点，-1代表自己是簇头
    nodes_is_alive = np.ones(node_nums, dtype=np.bool)  # True代表存活；False代表死亡

    # 迭代
    # 每轮存活节点数,返回来一个给定形状和类型的用0填充的数组
    alive = np.zeros(mat_iters)
    remain_energy = np.zeros(mat_iters)  # 每轮节点总能量
    distance_broad = bound * (2**0.5)
    for i in range(mat_iters):
        # 计算能量总合和存活节点数
        remain_energy[i] = nodes_power[nodes_is_alive].sum()  # 存活节点的能量总和
        alive[i] = np.sum(nodes_is_alive)  # 存活的节点数
        # 簇头选举
        not_cluster_idx = (~nodes_is_selected) & nodes_is_alive  # 存活且不是簇头的节点
        # 选取随机数小于等于阈值，则为簇头
        selected_idx = nodes_cluster_p[not_cluster_idx] <= (
            p / (1 - p * (i % round(1 / p)))
        )
        # 被选中的簇头
        if selected_idx.any():
            selected_cluster_idx = np.where(not_cluster_idx)[0][
                selected_idx
            ]  # 从不是簇头的节点中被选中的簇头
            nodes_is_normal[selected_cluster_idx] = False  # 节点类型为簇头
            nodes_is_selected[selected_cluster_idx] = True  # 标记为被选过簇头
            # nodes_cluster[selected_cluster_idx] = -1  # 自己是簇头
            # 广播自己成为簇头
            if distance_broad > d0:
                # 多径衰落信道能量消耗
                nodes_power[selected_cluster_idx] -= control_packet_length * (
                    E_consume + Emp * distance_broad**4
                )
            else:
                # 自由空间传输能量消耗
                nodes_power[selected_cluster_idx] -= control_packet_length * (
                    E_consume + Efs * distance_broad**2
                )
        not_selected_idx = ~selected_idx  # 不是簇头的节点
        if not_selected_idx.any():
            # 节点类型为普通
            nodes_is_normal[np.where(not_cluster_idx)[0][not_selected_idx]] = True
        # 判断最近的簇头结点，加入这个簇，如何去判断，采用距离矩阵
        distances = np.empty((node_nums, node_nums))  # 大小: n x n
        distances[:] = np.inf  # 初始值为无穷
        not_selected_idx = (
            ~nodes_is_selected
        ) & nodes_is_alive  # 存活且没有被选中的节点
        cluster_idx = (~nodes_is_normal) & nodes_is_alive  # 存活且是簇头的节点
        # 计算 存活且没有被选中的节点 到 存活且是簇头的节点 的距离
        distances[np.ix_(not_selected_idx, cluster_idx)] = np.linalg.norm(
            nodes_x_y[not_selected_idx][:, None] - nodes_x_y[cluster_idx], axis=2
        )
        # 找到距离簇头最近的簇成员节点
        cluster_min_idx = distances[not_selected_idx].argmin(axis=1)
        # 没有被选择的节点到簇头最近的距离
        dist = distances[not_selected_idx, cluster_min_idx]
        # 加入这个簇
        nodes_cluster[not_selected_idx] = cluster_min_idx
        d_less_d0 = dist < d0  # d < d0
        if d_less_d0.any():
            # 自由空间传输能量消耗
            nodes_power[np.where(not_selected_idx)[0][d_less_d0]] -= (
                control_packet_length * (E_consume + Efs * dist[d_less_d0] ** 2)
            )
        d_ge_d0 = ~d_less_d0  # d >= d0
        if d_ge_d0.any():
            # 多径衰落信道能量消耗
            nodes_power[np.where(not_selected_idx)[0][d_ge_d0]] -= (
                control_packet_length * (E_consume + Emp * dist[d_ge_d0] ** 4)
            )
        # 接收簇头发来的广播的能量消耗
        nodes_power[not_selected_idx] -= E_consume * control_packet_length
        # 对应簇头接收确认加入的能量消耗
        nodes_power[cluster_min_idx] -= E_consume * control_packet_length

        # 簇头接受发送，簇成员发送
        cluster_idx = (~nodes_is_normal) & nodes_is_alive  # 存活的簇头节点
        # 簇头接收普通节点发来的数据
        nodes_power[cluster_idx] -= (E_consume + ED) * packet_length
        # 簇头节点向基站发送数据
        cluster_base_distances = np.linalg.norm(
            nodes_x_y[cluster_idx] - base_station, axis=1
        )  # 簇头与基站的距离
        c_less_d0 = cluster_base_distances < d0  # d < d0
        # 能量消耗
        if c_less_d0.any():
            # 自由空间传输
            nodes_power[np.where(cluster_idx)[0][c_less_d0]] -= packet_length * (
                E_consume + ED + Efs * (cluster_base_distances[c_less_d0] ** 2)
            )
        c_ge_d0 = ~c_less_d0  # d >= d0
        if c_ge_d0.any():
            # 多径衰落信道
            nodes_power[np.where(cluster_idx)[0][c_ge_d0]] -= packet_length * (
                E_consume + ED + Emp * (cluster_base_distances[c_ge_d0] ** 4)
            )
        normal_idx = nodes_is_normal & nodes_is_alive  # 存活的普通节点
        # 普通节点向簇头发数据
        normal_cluster_distances = distances[normal_idx, nodes_cluster[normal_idx]]
        n_less_d0 = normal_cluster_distances < d0  # normal < d0
        if n_less_d0.any():
            # 自由空间传输
            nodes_power[np.where(normal_idx)[0][n_less_d0]] -= packet_length * (
                E_consume + ED + Efs * normal_cluster_distances[n_less_d0] ** 2
            )
        n_ge_d0 = ~n_less_d0  # normal >= d0
        if n_ge_d0.any():
            # 多径衰落信道
            nodes_power[np.where(normal_idx)[0][n_ge_d0]] -= packet_length * (
                E_consume + ED + Emp * normal_cluster_distances[n_ge_d0] ** 4
            )
        if alive[i] == 0:  # 若无节点存活则退出
            break
        nodes_is_alive[nodes_power < 0] = False  # 能量小于0的节点死亡
        nodes_cluster_p = rng.random(node_nums)  # 更新簇节点选取概率
    return alive, remain_energy


def benchmark():
    start = time.perf_counter()
    alive, remain_energy = leach()
    print(f"LEACH运行时间:{time.perf_counter() - start:.5f} s")
    plt.plot(alive, label="LEACH")
    plt.legend()
    plt.xlabel("轮数")
    plt.ylabel("存活节点数")

    plt.figure()
    plt.plot(remain_energy, label="LEACH")
    plt.legend()
    plt.xlabel("轮数")
    plt.ylabel("系统总能量")

    plt.figure()
    hist = np.histogram(alive, bins=(0, 20, 40, 100, 200, 201))
    print(hist)
    plt.bar(hist[1][:-1] / 2, hist[0], width=10)
    plt.xticks([0, 10, 20, 50, 100])
    plt.show()


if __name__ == "__main__":
    benchmark()
