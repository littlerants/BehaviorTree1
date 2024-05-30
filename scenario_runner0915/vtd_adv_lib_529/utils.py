
import math
from shapely.geometry import LineString


# 坐标系旋转操作 逆时针旋转theta
def rotate_operate(x,y,theta):
    tmpx = x
    tmpy = y

    x=  tmpx*math.cos(theta) - tmpy*math.sin(theta)
    y=  tmpy*math.cos(theta) + tmpx*math.sin(theta)   
    return [x,y]

# 检查并计算两个road是否相交，若相交，求出相交点
def check_lane_intersection(lane1_center, lane1_width, lane2_center, lane2_width):
    # 将中心线坐标转换为多边形
    lane1_polygon = LineString([(x, y) for x, y in lane1_center])
    lane2_polygon = LineString([(x, y) for x, y in lane2_center])

    # 使用buffer方法增加车道宽度
    lane1_polygon_buffered = lane1_polygon.buffer(lane1_width / 2, cap_style=2)
    lane2_polygon_buffered = lane2_polygon.buffer(lane2_width / 2, cap_style=2)
    intersection = False
    intersection_center = None
    if not (lane1_polygon_buffered.is_valid and lane2_polygon_buffered.is_valid):
        pass
    else:
        if lane1_polygon_buffered.intersects(lane2_polygon_buffered):
            # 计算相交区域
            intersection_area = lane1_polygon_buffered.intersection(lane2_polygon_buffered)
            if not intersection_area.is_empty:
                intersection = True
                intersection_center = list(intersection_area.centroid.coords)

    return intersection, intersection_center

def trans2angle(x, y, theta,ratate = 1):
    theta = ratate*theta
    tmp_x = x
    tmp_y = y
    new_pos_x = tmp_x * math.cos(theta) + tmp_y * math.sin(theta)
    new_pos_y = tmp_y * math.cos(theta) - tmp_x * math.sin(theta)
    print("new value:", new_pos_x, new_pos_y)
    return new_pos_x,new_pos_y