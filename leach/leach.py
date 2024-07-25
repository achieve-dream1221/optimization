import numpy as np
import matplotlib.pyplot as plt


def imp_leach():
    seed = None
    rng = np.random.default_rng(seed)
    node_nums = 200  # 节点数量
    mat_iters = 2000  # 迭代次数
    bound = 100  # 边界
    sink = (50, 125)  # 基站
    p = 0.08  # 簇头概率,未计算最佳簇头数量
    # 传输模型参数
    E_elec = 5e-8  # 单位bit数据发送、接收所消耗的能量
    Efs = 1e-11  # 自由空间传输
    Emp = 1.3e-15  # 多径衰落信道
    ED = 5e-9  # 能量消耗常量（Energy Dissipation）,传输数据时所消耗的能量
    d0 = 87  # 信道切换阈值 ？
    packet_length = 4000  # 数据包大小
    control_packet_length = 100  # 控制数据包大小
    E0 = 0.5
    E_min = 0.001  # 节点存活所需的最小能量
    radius = 15  # 初始通信距离
    # 节点随机分布
    nodes = np.empty((node_nums, 12))
    nodes_type = np.empty(node_nums, dtype='<U1')
    nodes_type[:] = 'N'  # type 进行选举簇头前先将所有节点设为普通节点, 'C': 当前节点为簇头节点
    nodes_selected = nodes_type.copy()  # selected 'O'：当选过簇头，N：没有
    nodes_neighbors = np.zeros((node_nums, node_nums))  # N 邻居节点集
    nodes_pre_neighbors = np.zeros((node_nums, node_nums))  # FN 前邻居节点集
    nodes_cluster_heads = np.zeros((node_nums, node_nums))  # CN 前邻居节点集
    nodes[:, [0, 1]] = rng.uniform(0, bound, (node_nums, 2))  # xd yd x, y坐标
    nodes[:, 2] = np.linalg.norm(nodes[:, [0, 1]] - sink, axis=1)  # d 节点与基站的距离
    nodes[:, 3] = radius  # Rc 节点通信距离
    nodes[:, 4] = rng.random(node_nums)  # temp_rand 随机数
    nodes[:, 5] = E0  # power 初始能量
    nodes[:, 6] = 0  # CH 保存普通节点的簇头节点，-1代表自己是簇头
    nodes[:, 7] = 1  # flag 1代表存活；0代表死亡
    nodes[:, 8] = 0  # Num_N 邻居节点集个数
    nodes[:, 9] = 0  # Num_FN 前邻居节点集个数
    nodes[:, 10] = 0  # Num_CN 前簇头节点集个数
    nodes[:, 11] = 0  # num_join 簇成员的个数
    # plt.scatter(sink[0], sink[1], marker="*", label="基站")
    # plt.scatter(nodes[:, 0], nodes[:, 1], marker="o", label="节点")
    # plt.legend()
    # plt.show()

    # 迭代
    alive_leach = np.zeros((mat_iters, 1))  # 每轮存活节点数,返回来一个给定形状和类型的用0填充的数组
    re_leach = alive_leach.copy()  # 每轮节点总能量
    for i in range(mat_iters):
        final_ch = []
        idx = nodes[:, 7] != 0
        re_leach[i] = nodes[idx, 5].sum()  # 更新总能量
        alive_leach[i] = np.sum(idx)  # 更新存活节点
        f = 0  # 判断是否没达到最大迭代次数mat_iters就退出了
        if alive_leach[i] == 0:
            stop = i
            f = 1
            break
        nodes[:, 3] = mat_iters * nodes[:, 5] / E0  # 节点的通信距离

        # 簇头选举
        flags = nodes[:, 7] != 0
        idx = ((nodes_selected == 'N') & flags)  # 是存活的普通节点
        flags_sum = np.sum(flags)
        if flags_sum == 0:  # 存活节点个数
            break
        alpha = np.ones(flags_sum) * 2  # 初始默认为2, 自由信道
        alpha[nodes[idx, 2] > d0] = 4  # 多经衰落 能量损失指数
        E_avg = nodes[idx, 5][flags].mean()  # 系统节点平均能量
        idx_1 = (nodes[idx, 4] <= (p / (1 - p * (i % round(1 / p)) * (nodes[idx, 5] / E_avg) ** (1 / alpha)))
                 and nodes[idx, 2] > nodes[idx, 3])  # xxxxxx且节点距离基站的距离大于Rc则可能成簇头
        count = np.sum(idx_1)  # 簇头个数
        nodes_type[idx][idx_1] = 'C'  # 节点类型为簇头
        nodes_selected[idx][idx_1] = 'O'  # 该节点标记'O'，说明当选过簇头
        nodes[idx[idx_1], 6] = -1  # 自己是簇头
        final_ch.extend(np.where(idx_1)[0])  # 索引加入簇头节点集合
        # 广播自己成为簇头
        distance_broad = nodes[idx[idx_1], 3] * (2 ** 0.5)
        idx_2 = distance_broad > d0
        nodes[idx[idx_1[idx_2]], 5] -= (control_packet_length * (E_elec + Emp * distance_broad[idx_2] ** 4)).sum()
        nodes[idx[idx_1[~idx_2]], 5] -= (control_packet_length * (E_elec + Emp * distance_broad[idx_2] ** 2)).sum()
        nodes_type[idx[~idx_1]] = 'N'  # 节点类型为普通

        # 计算邻居节点集合
        for a in range(node_nums):
            cnt = 0
            for b in range(node_nums):
                if a != b:
                    dist = np.linalg.norm(nodes[a, [0, 1]] - nodes[b, [0, 1]])
                    if dist <= nodes[a, 3]:
                        nodes_neighbors[a][cnt] = b
                        cnt += 1

        # 计算前邻节点集，更近邻居
        for a in range(node_nums):
            cnt = 0
            for b in range(nodes[i,]):
                if a != b:
                    dist = np.linalg.norm(nodes[a, [0, 1]] - nodes[b, [0, 1]])
                    if dist <= nodes[a, 3]:
                        nodes_neighbors[a][cnt] = b
                        cnt += 1


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
    radius = 15  # 初始通信距离
    # 节点随机分布
    nodes = np.empty((node_nums, 12))
    nodes_type = np.empty(node_nums, dtype='<U1')
    nodes_type[:] = 'N'  # type 进行选举簇头前先将所有节点设为普通节点, 'C': 当前节点为簇头节点
    nodes_selected = nodes_type.copy()  # selected 'O'：当选过簇头，N：没有
    # nodes_neighbors = np.zeros((node_nums, node_nums))  # N 邻居节点集
    # nodes_pre_neighbors = np.zeros((node_nums, node_nums))  # FN 前邻居节点集
    # nodes_cluster_heads = np.zeros((node_nums, node_nums))  # CN 前邻居节点集
    nodes[:, [0, 1]] = rng.uniform(0, bound, (node_nums, 2))  # xd yd x, y坐标
    # nodes[:, 2] = np.linalg.norm(nodes[:, [0, 1]] - base_station, axis=1)  # d 节点与基站的距离
    nodes[:, 3] = radius  # Rc 节点通信距离
    nodes[:, 4] = rng.random(node_nums)  # temp_rand 随机数
    nodes[:, 5] = E0  # power 初始能量
    nodes[:, 6] = 0  # CH 保存普通节点的簇头节点，-1代表自己是簇头
    nodes[:, 7] = 1  # flag 1代表存活；0代表死亡
    # nodes[:, 8] = 0  # Num_N 邻居节点集个数
    # nodes[:, 9] = 0  # Num_FN 前邻居节点集个数
    # nodes[:, 10] = 0  # Num_CN 前簇头节点集个数
    # nodes[:, 11] = 0  # num_join 簇成员的个数

    # 迭代
    alive_leach = np.empty(mat_iters)  # 每轮存活节点数,返回来一个给定形状和类型的用0填充的数组
    re_leach = np.empty(mat_iters)  # 每轮节点总能量
    distance_broad = bound * (2 ** 0.5)
    for i in range(mat_iters):
        # 计算能量总合和存活节点数
        idx = nodes[:, 7] != 0
        re_leach[i] = nodes[idx, 5].sum()
        alive_leach[i] = np.sum(idx)
        # 簇头选举
        idx = ((nodes_selected == 'N') & idx)  # 是存活的普通节点
        idx_1 = nodes[idx, 4] <= (p / (1 - p * (i % round(1 / p))))  # 选取随机数小于等于阈值，则为簇头
        if idx_1.any():
            indices = np.where(idx)[0][idx_1]
            nodes_type[indices] = 'C'  # 节点类型为蔟头
            nodes_selected[indices] = '0'  # 该节点标记'O'，说明当选过簇头
            nodes[indices, 6] = -1  # 自己是簇头
            # 广播自己成为簇头
            if distance_broad > d0:
                nodes[indices, 5] -= (control_packet_length * (E_consume + Emp * distance_broad ** 4))
            else:
                nodes[indices, 5] -= (control_packet_length * (E_consume + Efs * distance_broad ** 2))
        idx_1 = ~idx_1
        if idx_1.any():
            nodes_type[np.where(idx)[0][idx_1]] = 'N'  # 节点类型为普通
        # 判断最近的簇头结点，加入这个簇，如何去判断，采用距离矩阵
        yy = np.zeros((node_nums, node_nums))
        length = np.empty_like(yy)
        length[:] = np.inf
        flags = nodes[:, 7] != 0
        idx = ((nodes_selected == 'N') & flags)
        idx_1 = (nodes_type == 'C') & flags
        length[np.ix_(idx, idx_1)] = np.linalg.norm(nodes[idx, :2][:, None] - nodes[idx_1, :2], axis=2)
        # 找到距离簇头最近的簇成员节点
        min_idx = length[idx].argmin(axis=1)
        dist = length[idx, min_idx]
        # 加入这个簇
        idx_1 = dist < d0
        if idx_1.any():
            nodes[idx, 5][idx_1] -= (control_packet_length * (E_consume + Efs * dist[idx_1] ** 2))
        idx_1 = ~idx_1
        if idx_1.any():
            nodes[idx, 5][idx_1] -= (control_packet_length * (E_consume + Emp * dist[idx_1] ** 4))
        nodes[idx, 6] = min_idx
        # 接收簇头发来的广播的消耗
        nodes[idx, 5] -= E_consume * control_packet_length
        # 对应簇头接收确认加入的消息
        nodes[min_idx, 5] -= E_consume * control_packet_length
        yy[np.ix_(idx, min_idx)] = 1

        # 簇头接受发送，簇成员发送
        idx = nodes[:, 7] != 0
        idx = ((nodes_type == 'C') & idx)  # 是存活的普通节点
        # 簇头接收普通节点发来的数据
        nodes[idx, 5] -= (E_consume + ED) * packet_length
        # 簇头节点向基站发送数据
        distance = np.linalg.norm(nodes[idx, :2] - base_station, axis=1)
        idx_1 = distance < d0
        if idx_1.any():
            nodes[np.where(idx)[0][idx_1], 5] -= packet_length * (E_consume + ED + Efs * (distance[idx_1] ** 2))
        idx_1 = ~idx_1
        if idx_1.any():
            nodes[np.where(idx)[0][idx_1], 5] -= packet_length * (E_consume + ED + Emp * (distance[idx_1] ** 4))
        idx = nodes[:, 7] != 0
        idx = (nodes_type != 'C') & idx  # 是存活的普通节点
        # 普通节点向簇头发数据

        distance = length[idx, nodes[idx, 6].astype(int)]
        idx_1 = distance < d0
        if idx_1.any():
            nodes[np.where(idx)[0][idx_1], 5] -= packet_length * (E_consume + ED + Efs * distance[idx_1] ** 2)
        idx_1 = ~idx_1
        if idx_1.any():
            nodes[np.where(idx)[0][idx_1], 5] -= packet_length * (E_consume + ED + Emp * distance[idx_1] ** 4)
        if alive_leach[i] == 0:  # 若无节点存活则退出
            break
        idx = nodes[:, 5] < 0
        nodes[idx, 7] = 0
        nodes[:, 4] = rng.random(node_nums)  # 节点取一个(0,1)的随机值，与p比较
    return alive_leach, re_leach


def benchmark():
    alive, remain_energy = leach()
    plt.plot(alive, label="LEACH")
    plt.legend()
    plt.xlabel("轮数")
    plt.ylabel("存活节点数")

    plt.figure()
    plt.plot(remain_energy, label="LEACH")
    plt.legend()
    plt.xlabel("轮数")
    plt.ylabel("系统总能量")

    # plt.figure()
    # hist = np.histogram(alive, bins=(0, 20, 40, 100, 200, 201))
    # print(hist)
    # plt.bar(hist[1][:-1] / 2, hist[0], width=10)
    # plt.xticks([0, 10, 20, 50, 100])
    plt.show()


if __name__ == '__main__':
    benchmark()
