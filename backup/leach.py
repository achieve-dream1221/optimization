"""
如果要动态显示节点分簇情况，可以节点数设置40，迭代次数设置200，不然运行速度很慢，但是这样仿真结果对比不明显。
如果不需要动态显示，可以把动态显示节点分簇情况的代码（两处）注释，把节点数 n=400，迭代次数 rmax=2000
"""

import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(threshold=2000)
# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class BaseStation:  # 定义基站类
    x = 0  # 位置
    y = 0


class SensorNode:  # 定义传感器节点
    xd = 0  # 位置
    yd = 0
    d = 0  # 节点距基站的距离
    Rc = 0  # 节点的通信距离
    temp_rand = 0  # rand为(0,1)的随机数T(n)
    type = "N"  # 节点种类N为普通节点，C为簇头节点
    selected = "N"  # N为没有当选过过簇头，O为当选过簇头
    power = 0  # 初始能量
    CH = 0  # 保存普通节点的簇头节点，-1代表自己是簇头
    flag = 1  # 1代表存活；0代表死亡
    N = []  # 邻居节点集，并没有定义大小
    Num_N = 0  # 邻居节点集个数
    FN = []  # 前邻节点集
    Num_FN = 0  # 前邻节点集个数
    CN = []  # 前簇头节点集
    Num_CN = 0  # 前簇头节点集个数
    num_join = 0


# 初始化参数，减小节点总数 n 和迭代次数 rmax，可以减少代码运行时间
random.seed(2)  # 固定随机种子，使每次随机位置相同
n = 200  # 节点总数
rmax = 2000  # 迭代次数
is_display = 0  # 是否动态显示节点分簇情况
if is_display:
    # 节点太多，会导致出图较慢，看起来较乱，这里设置n = 40, rmax=400演示效果
    n = 40
    rmax = 200
xm = 100  # x轴范围
ym = 100  # y轴范围
sink = BaseStation()  # 基站
sink.x = 50  # 基站x轴
sink.y = 125  # 基站y轴
p = 0.08  # 簇头概率,未计算最佳簇头数量
# 传输模型参数
Eelec = 50 * (10 ** (-9))  # 单位bit数据发送、接收所消耗的能量
Efs = 10 * (10 ** (-12))  # 自由空间传输
Emp = 0.0013 * (10 ** (-12))  # 多径衰落信道
ED = 5 * (10 ** (-9))  # 能量消耗常量（Energy Dissipation）   传输数据时所消耗的能量
d0 = 87  # 信道切换阈值 ？
packetLength = 4000  # 数据包大小
ctrPacketLength = 100  # 控制数据包大小
E0 = 0.5  # 初始能量
Emin = 0.001  # 节点存活所需的最小能量
Rmax = 15  # 初始通信距离

## 节点随机分布
fig1 = plt.figure(dpi=80)
plt.grid(linestyle="dotted")
Node = []  # 节点集
plt.scatter(sink.x, sink.y, marker="*", s=200)
for i in range(n):
    node = SensorNode()
    node.xd = random.random() * xm
    node.yd = random.random() * ym  # 随机产生100个点
    node.d = (
        (node.xd - sink.x) ** 2 + (node.yd - sink.y) ** 2
    ) ** 0.5  # 节点距基站的距离
    node.Rc = Rmax  # 节点的通信距离
    node.temp_rand = random.random()  # rand为(0,1)的随机数
    node.type = "N"  # 进行选举簇头前先将所有节点设为普通节点, 'C': 当前节点为簇头节点
    node.selected = "N"  # 'O'：当选过簇头，N：没有
    node.power = E0  # 初始能量
    node.CH = 0  # 保存普通节点的簇头节点，-1代表自己是簇头
    node.flag = 1  # 1代表存活；0代表死亡
    node.N = [0 for _ in range(n)]  # 邻居节点集
    node.Num_N = 0  # 邻居节点集个数
    node.FN = [0 for _ in range(n)]  # 前邻节点集
    node.Num_FN = 0  # 前邻节点集个数
    node.CN = [0 for _ in range(n)]  # 前簇头节点集
    node.Num_CN = 0  # 前簇头节点集个数
    Node.append(node)
    plt.scatter(node.xd, node.yd, marker="o")
plt.legend(["基站", "节点"])
plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 15})  # 字体设置
plt.ylabel("y", fontdict={"family": "Times New Roman", "size": 15})
plt.show(block=False)
# plt.close()

# save data
flag = 1

################IMP_LEACH##################
# 迭代
alive_ima_leach = np.zeros(
    (rmax, 1)
)  # 每轮存活节点数,返回来一个给定形状和类型的用0填充的数组,rmax行1列
re_ima_leach = np.zeros((rmax, 1))  # 每轮节点总能量
for r in range(rmax):
    final_CH = []
    for i in range(n):
        if Node[i].flag != 0:
            re_ima_leach[r] = re_ima_leach[r] + Node[i].power  # 更新总能量
            alive_ima_leach[r] = alive_ima_leach[r] + 1  # 更新存活节点
    f = 0  # 判断是否没达到最大迭代次数rmax就退出了
    if alive_ima_leach[r] == 0:
        stop = r
        f = 1
        break
    for i in range(n):
        Node[i].temp_rand = random.random()  # 节点取一个(0,1)的随机值，与p比较
        Node[i].Rc = Rmax * Node[i].power / E0  # 节点的通信距离
        Node[i].CH = 0  # 保存普通节点的簇头节点，-1代表自己是簇头
        Node[i].N = np.zeros(n)  # 邻居节点集
        Node[i].Num_N = 0  # 邻居节点集个数
        Node[i].FN = np.zeros(n)  # 前邻节点集
        Node[i].Num_FN = 0  # 前邻节点集个数
        Node[i].CN = np.zeros(n)  # 前簇头节点集
        Node[i].Num_CN = 0  # 前簇头节点集个数
        Node[i].num_join = 1  # 簇成员的个数
    # 簇头选举
    count = 0  # 簇头个数
    for i in range(n):
        if Node[i].selected == "N" and Node[i].flag != 0:  # 是存活的普通节点
            if Node[i].d > d0:  # 多经衰落
                alpha = 4  # 能量损失指数
            else:  # 自由信道
                alpha = 2
            Eavg = 0  # 系统节点平均能量
            m = 0  # 存活节点个数

            for j in range(n):  # 计算系统节点平均能量
                if Node[i].flag != 0:
                    Eavg = Eavg + Node[i].power
                    m = m + 1
            if m != 0:
                Eavg = Eavg / n
            else:
                break

            if (
                Node[i].temp_rand
                <= (
                    p
                    / (
                        1
                        - p * (r % round(1 / p)) * (Node[i].power / Eavg) ** (1 / alpha)
                    )
                )
                and Node[i].d > Node[i].Rc
            ):  # xxxxxx且节点距离基站的距离大于Rc则可能成簇头
                Node[i].type = "C"  # 节点类型为簇头
                Node[i].selected = "O"  # 该节点标记'O'，说明当选过簇头
                Node[i].CH = -1  # 自己是簇头
                count = count + 1  # 簇头个数加一
                final_CH.append(i)  # 索引加入簇头节点集合
                # 广播自己成为簇头
                distanceBroad = (Node[i].Rc ** 2 + Node[i].Rc ** 2) ** 0.5
                if distanceBroad > d0:  # 发送能耗
                    Node[i].power = Node[i].power - (
                        Eelec * ctrPacketLength
                        + Emp * ctrPacketLength * (distanceBroad**4)
                    )
                else:
                    Node[i].power = Node[i].power - (
                        Eelec * ctrPacketLength
                        + Efs * ctrPacketLength * distanceBroad**2
                    )
            else:
                Node[i].type = "N"  # 节点类型为普通
    # 计算邻居节点集合
    for i in range(n):
        cnt = 0
        for j in range(n):
            if i != j:
                dist = (
                    (Node[i].xd - Node[j].xd) ** 2 + (Node[i].yd - Node[j].yd) ** 2
                ) ** 0.5
                if dist < Node[i].Rc:
                    cnt = cnt + 1
                    Node[i].N[cnt] = j
                    # if len(Node[i].N)<cnt:
                    #     Node[i].N.append(j)
                    # else:
                    #     Node[i].N[cnt-1] = j
            if j == n:
                Node[i].Num_N = cnt
    # 计算前邻节点集，更近邻居
    for i in range(n):
        cnt = 0
        for j in range(Node[i].Num_N):
            ne = Node[i].N[j]  # 获取邻居节点索引
            if Node[ne].d < Node[i].d:  # 如果邻居节点距离基站更近
                cnt = cnt + 1
                Node[i].FN[cnt] = ne
            if j == Node[i].Num_N:
                Node[i].Num_FN = cnt
    # 计算前簇头节点集
    for i in range(count):
        cnt = 0
        for j in range(Node[i].Num_FN):
            fne = Node[final_CH[i]].FN[j]  # 簇头的前邻居节点索引
            if fne != 0 and Node[fne].d < Node[final_CH[i]].d and Node[fne].CH == -1:
                cnt = cnt + 1
                Node[final_CH[i]].CN[cnt] = fne
            if j == Node[i].Num_FN:
                Node[final_CH[i]].Num_CN = cnt
    # 加入簇
    for i in range(n):
        if Node[i].type == "N" and Node[i].power > 0:
            E = np.zeros(count)  # count为簇头数
            for j in range(count):
                dist = (
                    (Node[i].xd - Node[final_CH[j]].xd) ** 2
                    + (Node[i].yd - Node[final_CH[j]].yd) ** 2
                ) ** 0.5
                if dist < Node[final_CH[j]].Rc:  # 满足条件1:在该簇头的通信范围内
                    E[j] = (Node[final_CH[j]].power - Emin) / Node[final_CH[j]].num_join
            if len(E) > 0:
                max_value, max_index = np.max(E), np.argmax(E)
            else:
                max_value, max_index = 0, 0
            # 节点发送加入簇的消息
            if len(final_CH) != 0:
                dist = (
                    (Node[i].xd - Node[final_CH[max_index]].xd) ** 2
                    + (Node[i].yd - Node[final_CH[max_index]].yd) ** 2
                ) ** 0.5
                if (
                    dist > Node[final_CH[max_index]].Rc
                ):  # 不满足条件1，选择最近的簇头加入
                    Length = np.zeros(count)
                    for j in range(count):
                        Length[j] = (
                            (Node[i].xd - Node[final_CH[j]].xd) ** 2
                            + (Node[i].yd - Node[final_CH[j]].yd) ** 2
                        ) ** 0.5
                    min_value, min_index = np.min(Length), np.argmin(Length)
                    Node[i].CH = final_CH[min_index]
                    ##################### 节点发送加入簇的消息       ？？？？为何没有用多经衰落，为何没有更新Rc
                    Node[i].power = Node[i].power - (
                        Eelec * ctrPacketLength + Efs * ctrPacketLength * (dist**2)
                    )
                    # 簇头接收消息
                    Node[final_CH[min_index]].power = (
                        Node[final_CH[min_index]].power - Eelec * ctrPacketLength
                    )
                    Node[final_CH[min_index]].num_join = (
                        Node[final_CH[min_index]].num_join + 1
                    )
                else:
                    # 节点发送加入簇的消息
                    Node[i].power = Node[i].power - (
                        Eelec * ctrPacketLength + Efs * ctrPacketLength * (dist**2)
                    )
                    # 簇头接收消息
                    Node[final_CH[max_index]].power = (
                        Node[final_CH[max_index]].power - Eelec * ctrPacketLength
                    )
                    Node[final_CH[max_index]].Rc = (
                        Rmax * Node[final_CH[max_index]].power / E0
                    )  # 通信距离随剩余能量而减少
                    Node[i].CH = final_CH[max_index]  # 以最大平均能量的节点为簇头
                    Node[final_CH[max_index]].num_join = (
                        Node[final_CH[max_index]].num_join + 1
                    )

    # 能量模型
    # 发送数据
    for i in range(n):
        if Node[i].flag != 0:
            if Node[i].type == "N" and Node[i].CH != 0:  # 普通节点    且有簇头
                dist = (
                    (Node[i].xd - Node[Node[i].CH].xd) ** 2
                    + (Node[i].yd - Node[Node[i].CH].yd) ** 2
                ) ** 0.5
                if dist > d0:
                    Node[i].power = Node[i].power - (
                        Eelec * packetLength + Emp * packetLength * (dist**4)
                    )
                else:
                    Node[i].power = Node[i].power - (
                        Eelec * packetLength + Efs * packetLength * (dist**2)
                    )
            else:  # 簇头节点
                #########################################为什么要加个ED，表示接受数据与发送数据消耗能量不一样？？为何接受数据只接受了一个packet
                Node[i].power = (
                    Node[i].power - (Eelec + ED) * packetLength
                )  # 簇头接收数据
                if Node[i].d <= Node[i].Rc:
                    Node[i].power = Node[i].power - (
                        Eelec * packetLength + Efs * packetLength * (Node[i].d ** 2)
                    )
                else:
                    if Node[i].Num_CN == 0:  # 没有比自己更靠近基站的簇头节点
                        if Node[i].d > d0:
                            Node[i].power = Node[i].power - (
                                Eelec * packetLength
                                + Emp * packetLength * (Node[i].d ** 4)
                            )
                        else:
                            Node[i].power = Node[i].power - (
                                Eelec * packetLength
                                + Efs * packetLength * (Node[i].d ** 2)
                            )
                    else:
                        # 选择中继节点
                        dis = np.zeros((Node[i].Num_CN, 1))
                        # 计算前簇头节点距基站的距离
                        for j in range(Node[i].Num_CN):
                            dis[j] = Node[Node[i].CN[j]].d
                        index = np.argsort(dis)  # 升序排序，且返回索引值
                        di = dis[index]
                        ######################### 中继转发   ？？？？为什么要给每一个前簇头节点都发
                        for j in range(Node[i].Num_CN):
                            Node[i].power = Node[i].power - di[j] / np.sum(di) * (
                                Eelec * packetLength
                                + Emp * packetLength * (di[Node[i].Num_CN + 1 - j] ** 2)
                            )
    for i in range(n):
        if Node[i].power < Emin:
            Node[i].flag = 0
    final_CH = []

    ########################################################################################
    # 动态显示节点分簇情况
    if is_display:
        if (r + 1) % 10 == 0:  # 每10次迭代画一次
            # fig2 = plt.figure(dpi=80)
            plt.cla()  # 清空画布
            plt.grid(linestyle="dotted")
            p1 = plt.scatter(sink.x, sink.y, marker="*", s=200)
            for i in range(n):
                if Node[i].type == "C":
                    p2 = plt.scatter(Node[i].xd, Node[i].yd, marker="^")
                    plt.plot([Node[i].xd, sink.x], [Node[i].yd, sink.y])
                else:
                    p3 = plt.scatter(Node[i].xd, Node[i].yd, marker="o")
                    plt.plot(
                        [Node[i].xd, Node[Node[i].CH].xd],
                        [Node[i].yd, Node[Node[i].CH].yd],
                    )
            plt.legend([p1, p2, p3], ["基站", "簇头节点", "普通节点"])
            plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 15})
            plt.ylabel("y", fontdict={"family": "Times New Roman", "size": 15})
            plt.title(f"IMP_LEACH：第{r + 1}次迭代", fontdict={"size": 15})
            plt.ion()  # 开启交互模式
            plt.show(block=False)
            plt.pause(0.15)  # 暂停0.15s

if is_display:
    plt.ioff()  # 关闭交互模式
    plt.show(block=False)
    ############################################################################################

if f == 0:
    stop = rmax

# load data.mat 节点复位
for i in range(n):
    # Node[i].temp_rand = random.random()           # rand为(0,1)的随机数
    Node[i].type = "N"  # 进行选举簇头前先将所有节点设为普通节点
    Node[i].selected = "N"  # 'O'：当选过簇头，N：没有
    Node[i].power = E0  # 初始能量
    Node[i].CH = 0  # 保存普通节点的簇头节点，-1代表自己是簇头
    Node[i].flag = 1  # 1代表存活；0代表死亡
    Node[i].N = [0 for _ in range(n)]  # 邻居节点集
    Node[i].Num_N = 0  # 邻居节点集个数
    Node[i].FN = [0 for _ in range(n)]  # 前邻节点集
    Node[i].Num_FN = 0  # 前邻节点集个数
    Node[i].CN = [0 for _ in range(n)]  # 前簇头节点集
    Node[i].Num_CN = 0  # 前簇头节点集个数
################LEACH##################
alive_leach = np.zeros((rmax, 1))  # 每轮存活节点数
re_leach = np.zeros((rmax, 1))  # 每轮节点总能量
for r in range(rmax):
    for i in range(n):  # 计算能量总合和存活节点数
        if Node[i].flag != 0:
            re_leach[r] = re_leach[r] + Node[i].power
            alive_leach[r] = alive_leach[r] + 1

    #    f = 0  # 判断是否没达到最大迭代次数rmax就退出了
    #    if alive_leach[r] == 0:#若无节点存活则退出
    #        stop = r
    #        f = 1
    #        break

    for i in range(n):
        Node[i].temp_rand = random.random()  # 节点取一个(0,1)的随机值，与p比较
    # 选簇头
    for i in range(n):
        if Node[i].selected == "N" and Node[i].flag != 0:
            # if  Node[i].type=='N' #只对普通节点进行选举，即已经当选簇头的节点不进行再选举
            if Node[i].temp_rand <= (
                p / (1 - p * (r % round(1 / p)))
            ):  # 选取随机数小于等于阈值，则为簇头
                Node[i].type = "C"  # 节点类型为蔟头
                Node[i].selected = "O"  # 该节点标记'O'，说明当选过簇头
                Node[i].CH = -1
                # 广播自成为簇头
                distanceBroad = (xm * xm + ym * ym) ** 0.5
                if distanceBroad > d0:
                    Node[i].power = Node[i].power - (
                        Eelec * ctrPacketLength
                        + Emp * ctrPacketLength * (distanceBroad**4)
                    )
                else:
                    Node[i].power = Node[i].power - (
                        Eelec * ctrPacketLength
                        + Efs * ctrPacketLength * (distanceBroad**2)
                    )
            else:
                Node[i].type = "N"  # 节点类型为普通
    # 判断最近的簇头结点，加入这个簇，如何去判断，采用距离矩阵
    yy = np.zeros((n, n))
    Length = np.zeros((n, n))
    for i in range(n):
        if Node[i].type == "N" and Node[i].flag != 0:
            for j in range(n):
                if (
                    Node[j].type == "C" and Node[j].flag != 0
                ):  # 计算普通节点到簇头的距离
                    Length[i, j] = (
                        (Node[i].xd - Node[j].xd) ** 2 + (Node[i].yd - Node[j].yd) ** 2
                    ) ** 0.5
                else:
                    Length[i, j] = float("inf")
            dist, index = (
                np.min(Length[i, :]),
                np.argmin(Length[i, :]),
            )  # 找到距离簇头最近的簇成员节点
            # 加入这个簇
            if Length[i, index] < d0:
                Node[i].power = Node[i].power - (
                    Eelec * ctrPacketLength
                    + Efs * ctrPacketLength * (Length[i, index] ** 2)
                )
            else:
                Node[i].power = Node[i].power - (
                    Eelec * ctrPacketLength
                    + Emp * ctrPacketLength * (Length[i, index] ** 4)
                )
            Node[i].CH = index
            # 接收簇头发来的广播的消耗
            Node[i].power = Node[i].power - Eelec * ctrPacketLength
            # 对应簇头接收确认加入的消息
            Node[index].power = Node[index].power - Eelec * ctrPacketLength
            yy[i, index] = 1
        else:
            Length[i, :] = float("inf")

    # 簇头接受发送，簇成员发送
    for i in range(n):
        if Node[i].flag != 0:
            if Node[i].type == "C":
                number = np.sum(yy[:, i])  # 统计簇头节点i的成员数量
                # 簇头接收普通节点发来的数据
                Node[i].power = Node[i].power - (Eelec + ED) * packetLength
                # 簇头节点向基站发送数据
                len = ((Node[i].xd - sink.x) ** 2 + (Node[i].yd - sink.y) ** 2) ** 0.5
                if len < d0:
                    Node[i].power = Node[i].power - (
                        (Eelec + ED) * packetLength + Efs * packetLength * (len**2)
                    )
                else:
                    Node[i].power = Node[i].power - (
                        (Eelec + ED) * packetLength + Emp * packetLength * len**4
                    )
            else:
                # 普通节点向簇头发数据
                len = Length[i, Node[i].CH]
                if len < d0:
                    Node[i].power = Node[i].power - (
                        Eelec * packetLength + Efs * packetLength * len**2
                    )
                else:
                    Node[i].power = Node[i].power - (
                        Eelec * packetLength + Emp * packetLength * len**4
                    )
    if alive_leach[r] == 0:  # 若无节点存活则退出
        stop = r
        f = 1
        break
    for i in range(n):
        if Node[i].power < 0:
            Node[i].flag = 0

    ########################################################################################
    # 动态显示节点分簇情况
    if is_display:
        if (r + 1) % 10 == 0:  # 每10次迭代画一次
            # fig2 = plt.figure(dpi=80)
            plt.cla()  # 清空画布
            plt.grid(linestyle="dotted")
            p1 = plt.scatter(sink.x, sink.y, marker="*", s=200)
            for i in range(n):
                if Node[i].type == "C":
                    p2 = plt.scatter(Node[i].xd, Node[i].yd, marker="^")
                    # print('hhh')
                    plt.plot([Node[i].xd, sink.x], [Node[i].yd, sink.y])
                else:
                    p3 = plt.scatter(Node[i].xd, Node[i].yd, marker="o")
                    plt.plot(
                        [Node[i].xd, Node[Node[i].CH].xd],
                        [Node[i].yd, Node[Node[i].CH].yd],
                    )
            plt.legend([p1, p2, p3], ["基站", "簇头节点", "普通节点"])
            plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 15})
            plt.ylabel("y", fontdict={"family": "Times New Roman", "size": 15})
            plt.title(f"LEACH：第{r + 1}次迭代", fontdict={"size": 15})
            plt.ion()  # 开启交互模式
            plt.show(block=False)
            plt.pause(0.15)  # 暂停0.15s

if is_display:
    plt.ioff()  # 关闭交互模式
    plt.show(block=False)
########################################################################################

if f == 0:
    stop = rmax
## 绘图显示
fig4 = plt.figure(dpi=80)
plt.plot(range(rmax), alive_ima_leach, c="r", linewidth=2)
plt.plot(range(rmax), alive_leach, c="b", linewidth=2)
plt.legend(["IMP_LEACH", "LEACH"])
plt.xlabel("轮数")
plt.ylabel("存活节点数")
fig5 = plt.figure(dpi=80)
plt.plot(range(rmax), re_ima_leach, c="r", linewidth=2)
plt.plot(range(rmax), re_leach, c="b", linewidth=2)
plt.legend(["IMP_LEACH", "LEACH"])
plt.xlabel("轮数")
plt.ylabel("系统总能量")
fig6 = plt.figure(dpi=80)
for r in range(rmax):
    if alive_ima_leach[r] >= n:
        a1 = r
    if alive_leach[r] >= n:
        a2 = r
    if alive_ima_leach[r] >= (1 - 0.1) * n:
        b1 = r
    if alive_leach[r] >= (1 - 0.1) * n:
        b2 = r
    if alive_ima_leach[r] >= (1 - 0.2) * n:
        c1 = r
    if alive_leach[r] >= (1 - 0.2) * n:
        c2 = r
    if alive_ima_leach[r] >= (1 - 0.5) * n:
        d1 = r
    if alive_leach[r] >= (1 - 0.5) * n:
        d2 = r
    if alive_ima_leach[r] > 0:
        e1 = r
    if alive_leach[r] > 0:
        e2 = r
y = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]]
y = np.array(y)
x = np.arange(5)
width = 0.36
plt.bar(x - width / 2, y[:, 0], width=width)
plt.bar(x + width / 2, y[:, 1], width=width)
plt.xticks(range(5), ["0", "10", "20", "50", "100"])
plt.legend(["IMP_LEACH", "LEACH"])
plt.xlabel("死亡比例")
plt.ylabel("循环轮数")

plt.show()
