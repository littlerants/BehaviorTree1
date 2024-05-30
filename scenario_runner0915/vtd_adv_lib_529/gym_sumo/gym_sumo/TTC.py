import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def line_equation_standard_form(point, angle):
    # point 是点的坐标 (x0, y0)，angle 是朝向的弧度
    x0, y0 = point
    # 计算斜率
    slope = np.tan(angle)
    # 标准形式方程的系数
    A = -slope
    B = 1
    C = -slope * x0 + y0
    return np.array([A, B, C])


def find_intersection_point(line1, line2):
    # line1 和 line2 分别表示两条直线的参数
    # 返回交点的坐标 (x, y)
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    # 构建方程组的系数矩阵和常数项向量
    coefficients_matrix = np.array([[A1, B1], [A2, B2]])
    constants_vector = np.array([C1, C2])

    try:
        # 求解方程组
        intersection_point = np.linalg.solve(coefficients_matrix, constants_vector)
        return np.array(intersection_point)
    except np.linalg.LinAlgError:
        # 处理无解的情况
        print("方程组无解。")
        return None


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    # 计算余弦相似度
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def is_overlapping(vertices1, vertices2):
    def get_axes(vertices):
        # 返回平行四边形的所有边的法向量作为分离轴
        return [(vertices[i][1] - vertices[i - 1][1], vertices[i - 1][0] - vertices[i][0]) for i in range(4)]

    def project(vertices, axis):
        # 投影平行四边形在轴上的投影范围
        dot_products = [np.dot(vertex, axis) for vertex in vertices]
        return min(dot_products), max(dot_products)

    # 获取两个平行四边形的轴
    axes = get_axes(vertices1) + get_axes(vertices2)

    for axis in axes:
        # 对每个轴进行投影
        proj1 = project(vertices1, axis)
        proj2 = project(vertices2, axis)

        # 如果投影没有重叠，那么两个平行四边形没有交集
        if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
            return False

    # 如果在所有轴上都有重叠，那么两个平行四边形相交
    return True


def angle_between_vectors(vector1, vector2):
    cosine_theta = cosine_similarity(vector1, vector2)
    theta_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    # 将弧度转换为角度
    theta_deg = np.degrees(theta_rad)
    # 调整角度范围为 0 到 180 度
    # if np.isnan(theta_deg):
    #     return 0.0
    # elif cosine_theta < 0:
    #     return 180.0 - theta_deg
    # else:
    #     return theta_deg
    return theta_deg


def TimeConditions(position, Obvelocity_1, ObCollisionPoint):
    DistanceOb_Collision = np.linalg.norm(position - ObCollisionPoint)
    T_Ob = DistanceOb_Collision / (Obvelocity_1 + 1e-5)
    return T_Ob


def plot_line_general_form(line, ax, x_range=(-5, 20), num_points=100, label=''):
    A, B, C = line
    if B != 0:
        # 不是垂直于坐标轴的情况
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = (C - A * x_values) / B
    elif A != 0:
        # 垂直于 y 轴的情况
        x_values = np.full(num_points, C / A)
        y_values = np.linspace(x_range[0], x_range[1], num_points)
    elif B != 0:
        # 垂直于 x 轴的情况
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.full(num_points, C / B)
    else:
        # A 和 B 都为 0，直线退化为点或不存在
        return

    # 绘制直线
    ax.plot(x_values, y_values, color='gray', linestyle='-',label=label)


class MotionState:
    def __init__(self, x, y, speed, psi_rad, width, length):
        self.x = x
        self.y = y
        self.speed = speed
        self.psi_rad = psi_rad
        self.width = width
        self.length = length
        self.polygon = self.polygon_xy_from_motionstate()
        self.lowleft = self.polygon[0].copy() # 车的右后角点
        self.lowright = self.polygon[1].copy()# 车的右前角点
        self.upright = self.polygon[2].copy()# 车的左前角点
        self.upleft = self.polygon[3].copy()# 车的左后角点
        self.left_line = line_equation_standard_form(self.upleft, self.psi_rad) #　车的左侧
        self.right_line = line_equation_standard_form(self.lowleft, self.psi_rad)# 车的右侧
        self.ego_on_bv_right = None

    def polygon_xy_from_motionstate(self):
        lowleft = (self.x - self.length / 2., self.y - self.width / 2.)
        lowright = (self.x + self.length / 2., self.y - self.width / 2.)
        upright = (self.x + self.length / 2., self.y + self.width / 2.)
        upleft = (self.x - self.length / 2., self.y + self.width / 2.)
        return rotate_around_center(np.array([lowleft, lowright, upright, upleft]),
                                    np.array([self.x, self.y]), yaw=self.psi_rad)

    def compute_intersection_point(self, other: 'MotionState'):
        alpha = angle_between_vectors(np.array([np.cos(self.psi_rad), np.sin(self.psi_rad)]),
                                      np.array([np.cos(other.psi_rad), np.sin(other.psi_rad)]))
        if alpha >= 179 or alpha <= 1:
            return None, None, None, None

        Q2 = find_intersection_point(self.left_line, other.left_line)
        Q1 = find_intersection_point(self.left_line, other.right_line)
        theta_deg = angle_between_vectors(Q2 - Q1, np.array([np.cos(self.psi_rad), np.sin(self.psi_rad)]))
        self.ego_on_bv_right = True if abs(theta_deg) < 1 else False
        Q4 = find_intersection_point(self.right_line, other.left_line)
        Q3 = find_intersection_point(self.right_line, other.right_line)
        return Q1, Q2, Q3, Q4

    def plot_collision_area(self, other: 'MotionState'):

        Q1, Q2, Q3, Q4 = self.compute_intersection_point(other)
        print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Q4: {Q4}')
        # 创建图形和轴
        fig, ax = plt.subplots()
        rect1 = matplotlib.patches.Polygon(self.polygon, closed=True, zorder=20, color='green')
        rect2 = matplotlib.patches.Polygon(other.polygon, closed=True, zorder=20, color='blue')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.text(self.x, self.y, 'ego', horizontalalignment='center', zorder=30)
        ax.text(other.x, other.y, 'bv', horizontalalignment='center', zorder=30)
        plot_line_general_form(self.left_line, ax)
        plot_line_general_form(self.right_line, ax)
        plot_line_general_form(other.left_line, ax)
        plot_line_general_form(other.right_line, ax)

        try:
            ax.text(Q1[0], Q1[1], 'Q1', horizontalalignment='center', zorder=30)
            ax.text(Q2[0], Q2[1], 'Q2', horizontalalignment='center', zorder=30)
            ax.text(Q3[0], Q3[1], 'Q3', horizontalalignment='center', zorder=30)
            ax.text(Q4[0], Q4[1], 'Q4', horizontalalignment='center', zorder=30)
        except TypeError:
            pass

        ax.text(self.upright[0], self.upright[1], 'A2', horizontalalignment='center', zorder=30)
        ax.text(self.upleft[0], self.upleft[1], 'B2', horizontalalignment='center', zorder=30)
        ax.text(self.lowleft[0], self.lowleft[1], 'C2', horizontalalignment='center', zorder=30)
        ax.text(self.lowright[0], self.lowright[1], 'D2', horizontalalignment='center', zorder=30)

        ax.text(other.upright[0], other.upright[1], 'A1', horizontalalignment='center', zorder=30)
        ax.text(other.upleft[0], other.upleft[1], 'B1', horizontalalignment='center', zorder=30)
        ax.text(other.lowleft[0], other.lowleft[1], 'C1', horizontalalignment='center', zorder=30)
        ax.text(other.lowright[0], other.lowright[1], 'D1', horizontalalignment='center', zorder=30)
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 20)
        # 设置图形的标题和标签
        ax.set_title('Matplotlib: Rotated Rectangle Example')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        filename = 'ego_on_bv_right' if self.ego_on_bv_right else 'ego_on_bv_left'
        plt.savefig(f'./{filename}.png')
        plt.show()

    def compute_ttc(self, other: 'MotionState'):
        TTCmin, TTCmax = None, None
        alpha = angle_between_vectors(np.array([np.cos(self.psi_rad), np.sin(self.psi_rad)]),
                                      np.array([np.cos(other.psi_rad), np.sin(other.psi_rad)]))
        if alpha < 1: # 同向行使
            dx = self.x - other.x
            dy = self.y - other.y
            diff_position = np.array([dx, dy])
            direction = np.array([np.cos(self.psi_rad), np.sin(self.psi_rad)])
            lat = abs(np.dot(diff_position, np.array([-direction[1], direction[0]])))
            if lat <= (self.width + other.width) / 2 + 0.1:
                temp_alpha = angle_between_vectors(diff_position, direction)

                if temp_alpha < 1: # 同向，主车在前
                    print('ego in front of bv')
                    TTCmin = TTCmax = None if other.speed <= self.speed else (
                            (np.linalg.norm(diff_position) - (self.length + other.length) / 2) / (other.speed - self.speed + 1e-5))
                else: # 背景车在前
                    print('bv in front of ego')
                    TTCmin = TTCmax = None if self.speed <= other.speed else (
                            (np.linalg.norm(diff_position) - (self.length + other.length) / 2) / (self.speed - other.speed + 1e-5))

            return (TTCmin, TTCmax) # 夹角小于１度，认为平行

        # 相向而行
        if alpha > 179:
            dx = self.x - other.x
            dy = self.y - other.y
            diff_position = np.array([dx, dy])
            direction = np.array([np.cos(self.psi_rad), np.sin(self.psi_rad)])
            lat = abs(np.dot(diff_position, np.array([-direction[1], direction[0]])))
            if lat <= (self.width + other.width) / 2 + 0.1:
                TTCmin = TTCmax = (np.linalg.norm(self.upright - other.lowright) /
                                   (self.speed + other.speed  +1e-5))

            return (TTCmin, TTCmax)

        Q1, Q2, Q3, Q4 = self.compute_intersection_point(other)
        A1, B1, C1, D1 = other.upright.copy(), other.upleft.copy(), other.lowleft.copy(), other.lowright.copy()
        A2, B2, C2, D2 = self.upright.copy(), self.upleft.copy(), self.lowleft.copy(), self.lowright.copy()
        print(f'A1: {A1}, B1: {B1}, C1: {C1}, D1: {D1}')
        print(f'A2: {A2}, B2: {B2}, C2: {C2}, D2: {D2}')
        egovelocity_2, Obvelocity_1 = self.speed, other.speed
        ego_arrived_intersetion = is_overlapping(self.polygon, [Q1, Q2, Q3, Q4])
        bv_arrived_intersetion = is_overlapping(other.polygon, [Q1, Q2, Q3, Q4])

        if not self.ego_on_bv_right:
            print('ego on bv left')
            A2, D2 = D2, A2
            B2, C2 = C2, B2
            A1, D1 = D1, A1
            B1, C1 = C1, B1
            Q1, Q4 = Q4, Q1
            Q3, Q2 = Q2, Q3
        else:
            print('ego on bv right')

        if alpha <= 89:
            ## 计算A2点碰撞
            T_ego_A2_Q1 = TimeConditions(Q1, egovelocity_2, A2)
            T_Ob_C1_Q1 = TimeConditions(Q1, Obvelocity_1, C1)
            T_Ob_D1_Q1 = TimeConditions(Q1, Obvelocity_1, D1)

            ## 计算C1点碰撞
            T_Ob_C1_Q3 = TimeConditions(Q3, Obvelocity_1, C1)  ## C1点离开可能发生碰撞区域
            T_ego_D2_Q3 = TimeConditions(Q3, egovelocity_2, D2)

            ## 计算D2点碰撞
            T_ego_D2_Q4 = TimeConditions(Q4, egovelocity_2, D2)
            T_Ob_B1_Q4 = TimeConditions(Q4, Obvelocity_1, B1)

            ## 计算D1点碰撞
            T_ego_B2_Q1 = TimeConditions(Q1, egovelocity_2, B2)

            ## 计算B2点碰撞
            T_ego_B2_Q2 = TimeConditions(Q2, egovelocity_2, B2)
            T_Ob_A1_Q2 = TimeConditions(Q2, Obvelocity_1, A1)

            ## 计算A1点碰撞
            T_ego_C2_Q4 = TimeConditions(Q4, egovelocity_2, C2)
            T_Ob_A1_Q4 = TimeConditions(Q4, Obvelocity_1, A1)

            if ego_arrived_intersetion:

                if T_ego_B2_Q1 > T_Ob_D1_Q1:
                    TTCmin, TTCmax = T_Ob_D1_Q1, T_Ob_D1_Q1

                if T_ego_B2_Q1 < T_Ob_D1_Q1 < T_Ob_A1_Q2 < T_ego_B2_Q2:
                    TTCmin, TTCmax = T_Ob_D1_Q1, T_Ob_A1_Q2

                if T_ego_C2_Q4 > T_Ob_A1_Q4 > T_Ob_A1_Q2 > T_ego_B2_Q2:
                    TTCmin = T_Ob_D1_Q1
                    TTCmax = T_Ob_A1_Q2

                return (TTCmin, TTCmax)

            elif bv_arrived_intersetion:

                if T_Ob_C1_Q1 > T_ego_A2_Q1 > T_Ob_D1_Q1:
                    TTCmin = T_ego_A2_Q1
                    TTCmax = T_ego_A2_Q1

                if T_Ob_C1_Q3 > T_ego_D2_Q3 > T_ego_A2_Q1 > T_Ob_C1_Q1:
                    TTCmin = T_ego_A2_Q1
                    TTCmax = T_ego_D2_Q3

                if T_Ob_B1_Q4 > T_ego_D2_Q4 > T_ego_D2_Q3 > T_Ob_C1_Q3:
                    TTCmin = T_ego_D2_Q3
                    TTCmax = T_ego_D2_Q4

                return (TTCmin, TTCmax)

            if T_Ob_C1_Q1 > T_ego_A2_Q1 > T_Ob_D1_Q1:
                TTCmin = T_ego_A2_Q1 if TTCmin is None else min(TTCmin, T_ego_A2_Q1)
                TTCmax = T_ego_A2_Q1 if TTCmax is None else min(TTCmax, T_ego_A2_Q1)

            if T_Ob_C1_Q3 > T_ego_D2_Q3 > T_ego_A2_Q1 > T_Ob_C1_Q1:
                TTCmin = T_ego_A2_Q1 if TTCmin is None else min(TTCmin, T_ego_A2_Q1)
                TTCmax = T_ego_D2_Q3 if TTCmax is None else min(TTCmax, T_ego_D2_Q3)

            if T_Ob_B1_Q4 > T_ego_D2_Q4 > T_ego_D2_Q3 > T_Ob_C1_Q3:
                TTCmin = T_ego_D2_Q3 if TTCmin is None else min(TTCmin, T_ego_D2_Q3)
                TTCmax = T_ego_D2_Q4 if TTCmax is None else min(TTCmax, T_ego_D2_Q4)

            if T_ego_B2_Q1 > T_Ob_D1_Q1 > T_ego_A2_Q1:
                TTCmin = T_Ob_D1_Q1 if TTCmin is None else min(TTCmin, T_Ob_D1_Q1)
                TTCmax = T_Ob_D1_Q1 if TTCmax is None else min(TTCmax, T_Ob_D1_Q1)

            if T_ego_B2_Q2 > T_Ob_A1_Q2 > T_Ob_D1_Q1 > T_ego_B2_Q1:
                TTCmin = T_Ob_D1_Q1 if TTCmin is None else min(TTCmin, T_Ob_D1_Q1)
                TTCmax = T_Ob_A1_Q2 if TTCmax is None else min(TTCmax, T_Ob_A1_Q2)

            if T_ego_C2_Q4 > T_Ob_A1_Q4 > T_Ob_A1_Q2 > T_ego_B2_Q2:
                TTCmin = T_Ob_D1_Q1 if TTCmin is None else min(TTCmin, T_Ob_D1_Q1)
                TTCmax = T_Ob_A1_Q2 if TTCmax is None else min(TTCmax, T_Ob_A1_Q2)

        elif 179 >= alpha >= 91:

            if ego_arrived_intersetion:
                T_ego_B2_Q2 = TimeConditions(Q2, egovelocity_2, B2)
                T_Ob_A1_Q2 = TimeConditions(Q2, Obvelocity_1, A1)
                if T_ego_B2_Q2 > T_Ob_A1_Q2:
                    TTCmin, TTCmax = T_Ob_A1_Q2, T_Ob_A1_Q2

                return (TTCmin, TTCmax)

            elif bv_arrived_intersetion:
                T_Ob_C1_Q3 = TimeConditions(Q3, Obvelocity_1, C1)
                T_ego_D2_Q3 = TimeConditions(Q3, egovelocity_2, D2)
                if T_Ob_C1_Q3 > T_ego_D2_Q3:
                    TTCmin, TTCmax = T_ego_D2_Q3, T_ego_D2_Q3

                return (TTCmin, TTCmax)

            ## 计算D2点碰撞
            T_Ob_C1_Q3 = TimeConditions(Q3, Obvelocity_1, C1)
            T_ego_D2_Q3 = TimeConditions(Q3, egovelocity_2, D2)
            T_Ob_D1_Q3 = TimeConditions(Q3, Obvelocity_1, D1)

            ## 计算D1点碰撞
            T_ego_A2_Q1 = TimeConditions(Q1, egovelocity_2, A2)
            T_Ob_D1_Q1 = TimeConditions(Q1, Obvelocity_1, D1)

            ## 计算A2点碰撞
            T_ego_A2_Q2 = TimeConditions(Q2, egovelocity_2, A2)
            T_Ob_A1_Q2 = TimeConditions(Q2, Obvelocity_1, A1)

            ## 计算A1点碰撞
            T_ego_B2_Q2 = TimeConditions(Q2, egovelocity_2, B2)

            if T_Ob_C1_Q3 > T_ego_D2_Q3 > T_Ob_D1_Q3:
                TTCmin = T_ego_D2_Q3 if TTCmin is None else min(TTCmin, T_ego_D2_Q3)
                TTCmax = T_ego_D2_Q3 if TTCmax is None else min(TTCmax, T_ego_D2_Q3)

            if T_Ob_D1_Q3 > T_ego_D2_Q3 and T_ego_A2_Q1 > T_Ob_D1_Q1 and T_ego_A2_Q1 > T_ego_D2_Q3 and T_Ob_D1_Q3 > T_Ob_D1_Q1:
                TTCmin = max(T_Ob_D1_Q1, T_ego_D2_Q3) if TTCmin is None else min(TTCmin, max(T_Ob_D1_Q1, T_ego_D2_Q3))
                TTCmax = min(T_ego_A2_Q1, T_Ob_D1_Q3) if TTCmax is None else min(TTCmax, min(T_ego_A2_Q1, T_Ob_D1_Q3))

            if T_Ob_D1_Q1 > T_ego_A2_Q1 and T_ego_A2_Q2 > T_Ob_A1_Q2 and T_Ob_D1_Q1 > T_Ob_A1_Q2 and T_ego_A2_Q2 > T_ego_A2_Q1:
                TTCmin = max(T_ego_A2_Q1, T_Ob_A1_Q2) if TTCmin is None else min(TTCmin, max(T_ego_A2_Q1, T_Ob_A1_Q2))
                TTCmax = min(T_Ob_D1_Q1, T_ego_A2_Q2) if TTCmax is None else min(TTCmax, min(T_Ob_D1_Q1, T_ego_A2_Q2))

            if T_ego_B2_Q2 > T_Ob_A1_Q2 > T_ego_A2_Q2:
                TTCmin = T_Ob_A1_Q2 if TTCmin is None else min(TTCmin, T_Ob_A1_Q2)
                TTCmax = T_Ob_A1_Q2 if TTCmax is None else min(TTCmax, T_Ob_A1_Q2)

        # 运动方向垂直
        elif 91 >= alpha >= 89:
            T_Ob_D1_Q1 = TimeConditions(Q1, Obvelocity_1, D1)
            T_ego_A2_Q1 = TimeConditions(Q1, egovelocity_2, A2)
            T_Ob_C1_Q3 = TimeConditions(Q3, Obvelocity_1, C1)
            T_ego_B2_Q2 = TimeConditions(Q2, egovelocity_2, B2)

            if ego_arrived_intersetion:
                if T_Ob_D1_Q1 < T_ego_B2_Q2:
                    TTCmin = T_Ob_D1_Q1
                    TTCmax = T_Ob_D1_Q1
                return (TTCmin, TTCmax)

            elif bv_arrived_intersetion:
                if T_ego_A2_Q1 < T_Ob_C1_Q3:
                    TTCmin = T_ego_A2_Q1
                    TTCmax = T_ego_A2_Q1

                return (TTCmin, TTCmax)

            if T_Ob_D1_Q1 < T_ego_A2_Q1 < T_Ob_C1_Q3:
                TTCmin = T_ego_A2_Q1 if TTCmin is None else min(TTCmin, T_ego_A2_Q1)
                TTCmax = T_ego_A2_Q1 if TTCmax is None else min(TTCmax, T_ego_A2_Q1)

            if T_ego_A2_Q1 < T_Ob_D1_Q1 < T_ego_B2_Q2:
                TTCmin = T_Ob_D1_Q1 if TTCmin is None else min(TTCmin, T_Ob_D1_Q1)
                TTCmax = T_Ob_D1_Q1 if TTCmax is None else min(TTCmax, T_Ob_D1_Q1)

        return (TTCmin, TTCmax)


if __name__ == "__main__":
    length1, length2, length3 = 5.0, 5.0, 5.0
    width1, width2, width3 = 2.0, 2.0, 2.0
    # # # 示例矩形框的信息 小于90度
    # x1, x2, x3 = 5.0, 1.0, 9.0
    # y1, y2, y3 = 1.0, 1.0, 1.0
    # angle1, angle2, angle3 = math.pi / 2, math.pi / 3,  math.pi * 2 / 3 # 朝向角度

    # # 示例矩形框的信息 大于90度
    # x1, x2, x3 = 5.0, 1.0, 9.0
    # y1, y2, y3 = 0.0, 9.0, 9.0
    # angle1, angle2, angle3 = math.pi / 2, -math.pi / 3,  -math.pi * 2 / 3 # 朝向角度

    # # 示例矩形框的信息 等于90度
    # x1, x2, x3 = 5.0, 1.0, 9.0
    # y1, y2, y3 = 2.5, 9.0, 9.0
    # angle1, angle2, angle3 = math.pi / 2, 0,  -math.pi # 朝向角度

    # 示例矩形框的信息 等于180度
    x1, x2, x3 = 6.0, 0.0, 12.0
    y1, y2, y3 = 14.2, 9.0, 9.0
    angle1, angle2, angle3 = -math.pi, 0,  -math.pi # 朝向角度

    veh1 = MotionState(x1, y1, 1,  angle1, width1, length1)
    veh2 = MotionState(x2, y2, 1, angle2, width2, length2)
    veh3 = MotionState(x3, y3, 1,  angle3, width3, length3)
    veh1.plot_collision_area(veh2)
    print(veh1.compute_ttc(veh2))
    veh1.plot_collision_area(veh3)
    print(veh1.compute_ttc(veh3))

