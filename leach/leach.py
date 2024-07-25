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
    # 节点随机分布
    nodes_type = np.ones(node_nums, dtype="<U1")  # type True: 普通节点, False: 当前节点为簇头节点
    nodes_type[:] = 'N'
    nodes_selected = nodes_type.copy()  # selected 'O'：当选过簇头，N：没有
    nodes = np.empty((node_nums, 2))
    nodes[:, [0, 1]] = rng.uniform(0, bound, (node_nums, 2))  # xd yd x, y坐标
    nodes_temp_rand = rng.random(node_nums)  # temp_rand 随机数
    nodes_power = np.ones(node_nums) * E0  # power 初始能量
    nodes_ch = np.zeros(node_nums, dtype=np.int32)  # CH 保存普通节点的簇头节点，-1代表自己是簇头
    nodes_flags = np.ones(node_nums, dtype=np.bool)  # flag 1代表存活；0代表死亡

    # 迭代
    alive = np.zeros(mat_iters)  # 每轮存活节点数,返回来一个给定形状和类型的用0填充的数组
    remain_energy = np.zeros(mat_iters)  # 每轮节点总能量
    distance_broad = bound * (2 ** 0.5)
    for i in range(mat_iters):
        # 计算能量总合和存活节点数
        remain_energy[i] = nodes_power[nodes_flags].sum()
        alive[i] = np.sum(nodes_flags)
        # 簇头选举
        idx = ((nodes_selected == 'N') & nodes_flags)  # 是存活的普通节点
        idx_1 = nodes_temp_rand[idx] <= (p / (1 - p * (i % round(1 / p))))  # 选取随机数小于等于阈值，则为簇头
        if idx_1.any():
            indices = np.where(idx)[0][idx_1]
            nodes_type[indices] = 'C'  # 节点类型为蔟头
            nodes_selected[indices] = '0'  # 该节点标记'O'，说明当选过簇头
            nodes_ch[indices] = -1  # 自己是簇头
            # 广播自己成为簇头
            if distance_broad > d0:
                nodes_power[indices] -= (control_packet_length * (E_consume + Emp * distance_broad ** 4))
            else:
                nodes_power[indices] -= (control_packet_length * (E_consume + Efs * distance_broad ** 2))
        idx_1 = ~idx_1
        if idx_1.any():
            nodes_type[np.where(idx)[0][idx_1]] = 'N'  # 节点类型为普通
        # 判断最近的簇头结点，加入这个簇，如何去判断，采用距离矩阵
        yy = np.zeros((node_nums, node_nums))
        length = np.empty_like(yy)
        length[:] = np.inf
        idx = ((nodes_selected == 'N') & nodes_flags)
        idx_1 = (nodes_type == 'C') & nodes_flags
        length[np.ix_(idx, idx_1)] = np.linalg.norm(nodes[idx][:, None] - nodes[idx_1], axis=2)
        # 找到距离簇头最近的簇成员节点
        min_idx = length[idx].argmin(axis=1)
        dist = length[idx, min_idx]
        # 加入这个簇
        idx_1 = dist < d0
        if idx_1.any():
            nodes_power[np.where(idx)[0][idx_1]] -= (control_packet_length * (E_consume + Efs * dist[idx_1] ** 2))
        idx_1 = ~idx_1
        if idx_1.any():
            nodes_power[np.where(idx)[0][idx_1]] -= (control_packet_length * (E_consume + Emp * dist[idx_1] ** 4))
        nodes_ch[idx] = min_idx
        # 接收簇头发来的广播的消耗
        nodes_power[idx] -= E_consume * control_packet_length
        # 对应簇头接收确认加入的消息
        nodes_power[min_idx] -= E_consume * control_packet_length
        yy[np.ix_(idx, min_idx)] = 1

        # 簇头接受发送，簇成员发送
        idx = ((nodes_type == 'C') & nodes_flags)  # 是存活的普通节点
        # 簇头接收普通节点发来的数据
        nodes_power[idx] -= (E_consume + ED) * packet_length
        # 簇头节点向基站发送数据
        distance = np.linalg.norm(nodes[idx] - base_station, axis=1)
        idx_1 = distance < d0
        if idx_1.any():
            nodes_power[np.where(idx)[0][idx_1]] -= packet_length * (E_consume + ED + Efs * (distance[idx_1] ** 2))
        idx_1 = ~idx_1
        if idx_1.any():
            nodes_power[np.where(idx)[0][idx_1]] -= packet_length * (E_consume + ED + Emp * (distance[idx_1] ** 4))
        idx = (nodes_type != 'C') & nodes_flags  # 是存活的普通节点
        # 普通节点向簇头发数据
        distance = length[idx, nodes_ch[idx].astype(int)]
        idx_1 = distance < d0
        if idx_1.any():
            nodes_power[np.where(idx)[0][idx_1]] -= packet_length * (E_consume + ED + Efs * distance[idx_1] ** 2)
        idx_1 = ~idx_1
        if idx_1.any():
            nodes_power[np.where(idx)[0][idx_1]] -= packet_length * (E_consume + ED + Emp * distance[idx_1] ** 4)
        if alive[i] == 0:  # 若无节点存活则退出
            break
        idx = nodes_power < 0
        nodes_flags[idx] = 0
        nodes_temp_rand = rng.random(node_nums)  # 节点取一个(0,1)的随机值，与p比较
    return alive, remain_energy


def benchmark():
    start = time.perf_counter()
    alive, remain_energy = leach()
    print(f"LEACH运行时间:{time.perf_counter() - start} s")
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


if __name__ == '__main__':
    benchmark()
