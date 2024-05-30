import copy
import random
import socket
import struct
import math
import os
import ctypes
import time
from collections import deque
import scipy.signal as signal
import numpy as np
import sys
import torch
from gymnasium import spaces
from shapely.geometry import Polygon #多边形
# from gym_sumo.algo.global_route_planner_vtd_xodr import GlobalRoutePlanner
# from gym_sumo.road.road import Road, RoadNetwork
# from gym_sumo.road.lane import LineType, PolyLaneFixedWidth
import py_trees
import carla
from  vtd_adv_lib_529.global1 import GLOBAL
from  vtd_adv_lib_529.model_namager import ModelManager
from  vtd_adv_lib_529.scenario import SCENARIO, PREPARE
from  vtd_adv_lib_529.object import OBJECT
from  vtd_adv_lib_529.utils import *
from  vtd_adv_lib_529.head import *
from  vtd_adv_lib_529.mobil import LATCHECK
from  vtd_adv_lib_529.config import CONFIG
import pandas as pd
import logging

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior,
)

def calculate_distance(other_actor, ego_vehicle):
    """
    Calculate the distance from the ego vehicle
    @param other_actor: other vehicle
    @param ego_vehicle: ego vehicle
    @return: float
    """
    return other_actor.get_location().distance(ego_vehicle.get_location())
def get_speed(vehicle, meters=False):
    """
    Compute speed of a vehicle in Km/h.

    Parameters
    ----------
    meters : bool
        Whether to use m/s (True) or km/h (False).

    vehicle : carla.vehicle
        The vehicle for which speed is calculated.

    Returns
    -------
    speed : float
        The vehicle speed.
    """
    vel = vehicle.get_velocity()
    vel_meter_per_second = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    return vel_meter_per_second if meters else 3.6 * vel_meter_per_second
import subprocess
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.DEBUG)



# VTD message 报文头处理
class HANDLE():
    def __init__(self,handle = [0,0,0,0,0,0]):
        # # (magicNo,version,headerSize,dataSize,frameNo,simTime)
        self.magicNo = handle[0]
        self.version = handle[1]
        self.headerSize = handle[2]
        self.dataSize = handle[3]
        self.frameNo = handle[4]
        self.simTime = handle[5]
    def update(self,handle):
        # # (magicNo,version,headerSize,dataSize,frameNo,simTime)
        self.magicNo = handle[0]
        self.version = handle[1]
        self.headerSize = handle[2]
        self.dataSize = handle[3]
        self.frameNo = handle[4]
        self.simTime = handle[5]
    def show(self):
        return ("(magicNo:{},version:{},headerSize: {},dataSize: {},frameNo: {},simTime: {})".format(self.magicNo,self.version, self.headerSize,self.dataSize, self.frameNo, self.simTime))



# VTD 管理类
class ADV_Manager(AtomicBehavior):
    # 速度阶梯查询表，用来计算对抗车预期加速度，假设车辆行驶速度范围0-30
    # def __init__(self, ego_actor=None, name="ADV_Manager"):

    def __init__(self, ego_actor=None, name="ADV_Manager"):
        super(ADV_Manager, self).__init__(name)
        self.args = CONFIG()
        self.lat_check = LATCHECK()
        self.model_manager =  ModelManager(self.args ,self.args.model_path)

        self.static_objects_config = {}
        self.static_objects = []

        self.wall_far = 70
        self.model_type = None
        # 强化学习模型动作空间中各个动作执行时间
        self.ACTIONS_DUR =  { 'LANE_LEFT': 20, 'IDLE': 0 , 'LANE_RIGHT': 20, 'FASTER': 10, 'SLOWER': 10,
                            "LEFT_1": 10, "LEFT_2":20, "RIGHT_1": 20, "RIGHT_2": 20,"BLANK":10}
        self.action_marking = ''
        # 各个动作预留时间
        self.keep_time = -1
        # 各个动作已用动作时间
        self.keep_time_index = -1
        # self.disappear_num = 0
        # 控制指令
        self.ctrl_signal = None
        # 安全距离
        self.safe_dis = 4
        # 真实距离
        self.dis = 0
        # carla
        self.client = CarlaDataProvider.get_client()
        self._map = CarlaDataProvider.get_map()
        self.world = CarlaDataProvider.get_world()
        self._tm_port = CarlaDataProvider.get_traffic_manager_port()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(self._tm_port)
        self.ego_vehicle = ego_actor
        # 初始化全局对象
        self.gl = GLOBAL()
        self.DEFAULT_TARGET_SPEEDS = np.linspace(0, 30, self.gl.acc_resolution)

    def create_static_objects(self,prepare):
        for i in range(len(self.prepare.wallpoints)):
            wall_lane_info = self.road.network.get_road_info(
                self.road.network.get_closest_lane_index(self.prepare.wallpoints[i])[:2] )
            wall_lane = self.road.network.get_lane(
                self.road.network.get_closest_lane_index(self.prepare.wallpoints[i]  ) )
            self.static_objects_config["static_wall_point_"+ str(i)] =  {"name":"static_wall_point","static":True, "id":-(i+1), "pos_x":prepare.wallpoints[i][0],
                    "pos_y":prepare.wallpoints[i][1],"lane":wall_lane, "lane_id":wall_lane_info["lane_id"],
             "roadId":wall_lane_info["road_id"], "pos_h":wall_lane_info["heading" ] }

            self.static_objects.append( OBJECT(static=True,id=-(i+1),pos_x=prepare.wallpoints[i][0],pos_y=prepare.wallpoints[i][1],\
                name="static_wall_point_"+ str(i),lane= wall_lane,lane_id=wall_lane_info["lane_id"],roadId=wall_lane_info["road_id"] ,pos_h=wall_lane_info["heading"] )  )

    # 获取地图中所有juction信息
    def juction_set(self):

        if len(self.junction_edge_list) == 0:
            for edge_id in self.map.graph.edges:
                edge = self.map.graph.get_edge_data(*edge_id)
                if len(edge['path']) == 0:
                    continue
                if edge['entry_waypoint'].junction_id is not None:
                    self.junction_edge_list.add(edge_id)
                    self.junction_road_info[edge_id] = (
                    edge['entry_waypoint'].road_id, edge['entry_waypoint'].section_id, edge['entry_waypoint'].lane_id)


    # 初始化road对象
    def make_road(self,map) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork(map)
        lane_id = set()
        for edge_id in map.graph.edges():
            edge = map.graph.get_edge_data(edge_id[0], edge_id[1])
            length, start_wpt, end_wpt, edge_type = edge.get('length'), edge.get('entry_waypoint'), edge.get(
                'exit_waypoint'), edge.get('type')
            if length > 1:
                center_points = [start_wpt]
                center_points.extend(edge.get('path'))
                center_points.append(end_wpt)
                center_lines = []
                for wpt in center_points:
                    center_lines.append([wpt.x, wpt.y])
                dis = np.linalg.norm(
                    [center_lines[-1][0] - center_lines[-2][0], center_lines[-1][1] - center_lines[-2][1]])
                if dis < 1e-4:
                    print("edge dis is 0!!!:",edge_id)
                lane_id.add(edge_id)
                net.add_lane(edge_id[0], edge_id[1],
                                PolyLaneFixedWidth(center_lines,
                                                start_wpt.width,
                                                lane_index=edge_id,
                                                line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE))   )

        return  Road(network=net, np_random=1, record_history=False)

    def get_sqrt(self,x,y):
        return math.sqrt(x**2 + y**2)

    def chek_in_juction(self,roadId,section, lane_id):
        in_juction = False
        # print("ADV---11111:",roadId,section, lane_id)
        lane_index = self.map.road_id_to_edge[  roadId  ][section][ lane_id ]
        if lane_index in self.junction_edge_list:
            in_juction = True
        return lane_index,  in_juction
    def update_info(self):
        """
        Update ego vehicle info
        Returns: None
        """
        ego_pos = self.ego_vehicle.get_transform()
        ego_speed = get_speed(self.ego_vehicle)
        objects = {}
        static_objects = []
        other_vehicles = []
        traffic_lights = []
        for other_vehicle in self.world.get_actors().filter('vehicle.*'):
            if calculate_distance(other_vehicle, self.ego_vehicle) < 50 and \
                    other_vehicle.attributes['role_name'] != "ego_vehicle":
                if get_speed(other_vehicle) > 0:
                    other_vehicles.append(other_vehicle)
                else:
                    static_objects.append(other_vehicle)
        for light in self.world.get_actors().filter('traffic.traffic_light*'):
            if calculate_distance(light, self.ego_vehicle) < 50:
                traffic_lights.append(light)
        for walker in self.world.get_actors().filter('walker.*'):
            if calculate_distance(walker, self.ego_vehicle) < 50:
                if get_speed(walker) > 0:
                    other_vehicles.append(walker)
                else:
                    static_objects.append(walker)
        for static_obj in self.world.get_actors().filter('*static*'):
            if calculate_distance(static_obj, self.ego_vehicle) < 20:
                static_objects.append(static_obj)

        objects["vehicles"] = other_vehicles
        objects["traffic_lights"] = traffic_lights
        objects["statics"] = static_objects
        return  other_vehicles,traffic_lights

    def get_local_location(self, vehicle, location) -> carla.Location:
        """将全局坐标系下的坐标转到局部坐标系下

        Args:
            location (Location): 待变换的全局坐标系坐标
        """
        res = np.array(vehicle.get_transform().get_inverse_matrix()).dot(
            np.array([location.x, location.y, location.z, 1])
        )
        return carla.Vector3D(x=res[0], y=res[1], z=res[2])
    # 初始化或更新目标对象
    def create_objs(self):

        objects, traffic_lights = self.update_info()
        for i in objects:
            self._tm.force_lane_change(i, False)
        ego_wp = self._map.get_waypoint(self.ego_vehicle.get_location())
        # 车辆朝向 左手坐标系
        pos_h = (-self.ego_vehicle.get_transform().rotation.yaw/180)*np.pi
        # 道路朝向
        hdg = (-ego_wp.transform.rotation.yaw/180)*np.pi
        # self.gl.objects_set 所要维护的目标列表
        # 若为0，则直接清空 self.gl.objects（用于存储所有目标对象）
        if len(self.gl.objects_set) == 0:
            self.gl.objects.clear()
        # 如果主车不为空，更新主车各个参数
        if self.gl.ego is not None:

            self.gl.ego.update(pos_x=self.ego_vehicle.get_location().x, pos_y=-self.ego_vehicle.get_location().y,
                               pos_h=pos_h, \
                               vx=self.ego_vehicle.get_velocity().x, vy=-self.ego_vehicle.get_velocity().y,
                               acc_x=self.ego_vehicle.get_acceleration().x, hdg=hdg, \
                               l=self.ego_vehicle.bounding_box.extent.x, w=self.ego_vehicle.bounding_box.extent.y,
                               acc_y=-self.ego_vehicle.get_acceleration().y, roadId=ego_wp.road_id,
                               obj_type=self.ego_vehicle.type_id, lane_offset=0, lane_id=ego_wp.lane_id, \
                               leftLaneId=ego_wp.get_left_lane().lane_id if ego_wp.get_left_lane() else 100,
                               rightLaneId=ego_wp.get_right_lane().lane_id if ego_wp.get_right_lane() else 100, \
                               light_state=traffic_lights,
                               wp=ego_wp
                               )
        # 否则初始化主车对象
        else:
            self.gl.ego = OBJECT(name=self.ego_vehicle.id, \
                                 id=self.ego_vehicle.id, pos_x=self.ego_vehicle.get_location().x, off_x=0,
                                 pos_y=-self.ego_vehicle.get_location().y,
                                 pos_h=-self.ego_vehicle.get_transform().rotation.yaw, \
                                 hdg=-ego_wp.transform.rotation.yaw, vx=self.ego_vehicle.get_velocity().x,
                                 vy=-self.ego_vehicle.get_velocity().y, roadId=ego_wp.road_id, \
                                 acc_x=self.ego_vehicle.get_acceleration().x, l=self.ego_vehicle.bounding_box.extent.x,
                                 w=self.ego_vehicle.bounding_box.extent.y, \
                                 acc_y=-self.ego_vehicle.get_acceleration().y, obj_type=self.ego_vehicle.type_id,
                                 lane_offset=0, \
                                 lane_id=ego_wp.lane_id,
                                 leftLaneId=ego_wp.get_left_lane().lane_id if ego_wp.get_left_lane() else 100,
                                 rightLaneId=ego_wp.get_right_lane().lane_id if ego_wp.get_right_lane() else 100,
                                 wp=ego_wp
                                 )
        # 遍历self.gl.fram_data['Objects'] 中探测到的目标，并初始化或更新这些目标
        
        # print("self.gl.fram_data:",self.gl.fram_data)
        for i in objects:
            if  i.id == self.ego_vehicle.id:
                continue
            location = self.get_local_location(self.ego_vehicle, i.get_location())
            i_wp = self._map.get_waypoint(i.get_location())
            pos_h = (-i.get_transform().rotation.yaw/180)* np.pi
            hdg = (-i_wp.transform.rotation.yaw/180)* np.pi
            if i.id in self.gl.objects_set :
                for j in self.gl.objects:
                    if i.id == j.id:
                        j.update(pos_x=location.x, pos_y=-location.y,
                                 pos_h=pos_h, hdg=hdg, \
                                 vx=i.get_velocity().x, vy=-i.get_velocity().y,
                                 acc_x=i.get_acceleration().x, l=i.bounding_box.extent.x, w=i.bounding_box.extent.y,
                                 acc_y=-i.get_acceleration().y, obj_type=i.type_id, \
                                 lane_offset=0, lane_id=i_wp.lane_id, roadId=i_wp.road_id, \
                                 leftLaneId=i_wp.get_left_lane().lane_id if i_wp.get_left_lane() else 100,
                                 rightLaneId=i_wp.get_right_lane().lane_id if i_wp.get_right_lane()  else 100,
                                 wp=i_wp
                                 )
            else:
                obj = OBJECT(id=i.id, pos_x=location.x, pos_y=-location.y, off_x=0,
                             pos_h=pos_h, hdg=hdg, \
                             vx=i.get_velocity().x, vy=-i.get_velocity().y,
                             acc_x=i.get_acceleration().x, l=i.bounding_box.extent.x, w=i.bounding_box.extent.y,
                             acc_y=-i.get_acceleration().y, obj_type=i.type_id, \
                             lane_offset=0, lane_id=i_wp.lane_id, roadId=i_wp.road_id, \
                             leftLaneId=i_wp.get_left_lane().lane_id if i_wp.get_left_lane() else 100,
                             rightLaneId=i_wp.get_right_lane().lane_id if i_wp.get_right_lane() else 100,
                             wp=i_wp
                             )

                obj.initial_state2ego = {"pos_x":obj.pos_x,"pos_y":obj.pos_y,"pos_h":obj.pos_h}
                self.gl.objects_set.append(obj.id)
                self.gl.objects+= [obj]

        # if len(self.static_objects) > 0:
        #     for i in self.static_objects:
        #         dis2ego = self.get_sqrt(self.static_objects_config[i.name]['pos_x'] - self.gl.ego.pos_x,\
        #                                 self.static_objects_config[i.name]['pos_y'] - self.gl.ego.pos_y)
        #         # print("dis2ego:",dis2ego)
        #         if dis2ego < 200 and  i.id not in self.gl.objects_set:
        #             # static_tmp = copy.deepcopy(i)
        #             self.gl.objects.append(i)
        #             self.gl.objects_set.append(i.id)
        #         i.update(pos_x=self.static_objects_config[i.name]['pos_x'],pos_y=self.static_objects_config[i.name]['pos_y'],\
        #                  vx=0,vy = 0,acc_x=0, acc_y=0,\
        #                  lane=self.static_objects_config[i.name]['lane'], lane_id=self.static_objects_config[i.name]['lane_id'], roadId=self.static_objects_config[i.name]['roadId'], \
        #                  pos_h=self.static_objects_config[i.name]['pos_h'],static = True
        #                  )
        #         i.trans_cood2(self.gl.ego, True, True, True, rotate=1)
        print("---------------------ego info-----------------------:")
        self.gl.ego.show()
        for i in self.gl.objects:
            print("----------------------------------111")
            i.show()
            i.global_trans_local(self.gl.ego,velocity_flag=True, acc_flag=True,rotate=-1)
            i.show()
            print("----------------------------------222")

    def get_sqrt(self,x,y):
        return math.sqrt(x**2 + y**2)


    def obj2ego_djirection(self,i):
        # 计算各个车辆与主车的方位 ，会根据这些方位制定对应的策略

        # 如果在主车前方3米，则标记为0
        threshold = 1.5
        ego_lane_id = self.gl.ego.lane_id
        if i.pos_x > 0 and i.lane_id  == ego_lane_id:
            i.direction_to_ego = 0
        if i.pos_x >0 and i.pos_y > threshold and  i.lane_id != ego_lane_id:
            i.direction_to_ego = 1
        if i.pos_x > 0 and i.pos_y < threshold and i.lane_id != ego_lane_id:
            i.direction_to_ego = 2
        if i.pos_x  < 0 and i.lane_id  == ego_lane_id:
            i.direction_to_ego = 3
        if i.pos_x <0 and i.pos_y > threshold and  i.lane_id != ego_lane_id:
            i.direction_to_ego = 4
        if i.pos_x < 0 and i.pos_y < threshold and i.lane_id != ego_lane_id:
            i.direction_to_ego = 5
        # threshold = 2
        # # front
        # if i.pos_x >= 0 and abs(i.pos_y) <= threshold :
        #     i.direction_to_ego = 0
        # # front left
        # elif i.pos_x >= 0 and i.pos_y > threshold:
        #     i.direction_to_ego = 1
        # # front right
        # elif i.pos_x >= 0 and i.pos_y < -threshold :
        #     i.direction_to_ego = 2
        # # back
        # elif i.pos_x < 0 and abs(i.pos_y) <= threshold:
        #     i.direction_to_ego = 3
        # # left back
        # elif i.pos_x < 0 and i.pos_y > threshold:
        #     i.direction_to_ego = 4
        # # right back
        # elif i.pos_x < 0 and i.pos_y < -threshold:
        #     i.direction_to_ego = 5
    def check_adv_enable(self):
        if len(self.prepare.adv_enable) == 0:
            return True
        for i in range(len(self.prepare.adv_enable)):
            rel_ab_pos_x = self.prepare.adv_enable[i][0] - self.gl.ego.pos_x
            rel_ab_pos_y = self.prepare.adv_enable[i][1] - self.gl.ego.pos_y
            rel_pos_x,rel_pos_y = trans2angle(rel_ab_pos_x,rel_ab_pos_y,self.gl.ego.pos_h)
            print("ADV--- adv rel_pos_x:",rel_pos_x)
            # if rel_pos_x > 0 and self.get_sqrt(rel_pos_x,rel_pos_y) < 5:
            if rel_pos_x < 5:
                self.gl.adv_flag = True
                # 如果到达使能点，但对抗车还为追赶上主车，则放弃对抗
                if self.gl.adv and self.gl.adv.pos_x < -5:
                    self.gl.adv_flag = False
                return  True
            return False

    def parseRDBMessage(self):

        #  创建对象
        self.create_objs()

        # 根据与主车的距离远近，对所有探测到的目标物进行排序
        self.gl.objects = sorted(self.gl.objects, key=lambda x: math.sqrt(x.pos_x**2 + x.pos_y**2) )
        for i in self.gl.objects:
            self.obj2ego_djirection(i)


        flag = {'index':0,"flg":False}
        # 将要从 self.objects中弹出的对象列表
        pop_index = []
        index = 0
        # 与主车最近的一辆车的index，若对抗车存在且不是最近的目标物，则此目标作为备选对抗车辆
        front_close_vec_index = -1

        show_log = 1
        if show_log:
           print("ADV---len(self.gl.objects):",len(self.gl.objects))
        if len(self.gl.objects) >0:
            # print("ADV---len objects:",len(self.gl.objects))g
            ego_vx,ego_vy = trans2angle(self.gl.ego.vx, self.gl.ego.vy,self.gl.ego.pos_h)
            for i in self.gl.objects:
                #
                real_vx = i.vx + ego_vx
                real_vy = i.vy + ego_vy
                if real_vx < 0.2 and ego_vx > 2:
                    index+=1
                    continue
                # print('distance:',self.get_sqrt(i.pos_x, i.pos_y))
                # 主车50米以内所有车辆
                if (i.name.find("static")== -1 or not i.static) and  self.get_sqrt(i.pos_x, i.pos_y) <90:
                    # 主车后30米以内所有车辆
                    if  i.pos_x > -50:
                        # print("ADV---i.pos_x > 3")
                        # i.show()
                        # print("ADV---self.gl.compete_time:",self.gl.compete_time)
                        # 选定一辆对抗车后，对抗时间持续 500 帧
                        if self.gl.compete_time < self.gl.compete_time_range :

                            if front_close_vec_index == -1:
                                # 获取距离主车最近的目标index
                                front_close_vec_index = index
                                # front_close_vec_index = -1
                            # 如果对抗时长还未用完，且当前目标物与上一帧目标物为同一目标物，则被优先选择
                            # if i.name ==  self.gl.last_compete_name:
                            if i.name ==  'New Player':
                                pop_index.append(index)
                                flag['flg'] = True
                                flag['index'] = index
                        # 对抗时间用完，则清空所有对抗信息，重新选择对抗目标
                        else:
                            # auto pilot

                            self.gl.compete_time = 0
                            self.gl.last_compete_name = ''
                            self.gl.last_compete_name_id = -1
                            print("ADV---comete time is over!!! free adv vecicle:",self.gl.adv.name)
                            self.gl.adv = None
                            return None
                elif i.name.find("static") != -1 and  self.get_sqrt(i.pos_x, i.pos_y) > self.wall_far:
                    pop_index.append(index)

                index += 1

        else:
            # print()
            self.gl.adv = None
            self.gl.compete_time = 0
            print("ADV---ego near no compete vecs !!!")
            return None

        if show_log:
           print("ADV---flag::",flag)
           print("ADV---front_close_vec_index:",front_close_vec_index)
        # 更新对抗车辆
        if flag['flg'] :
            if self.gl.adv is not  None and  self.gl.adv.name == self.gl.objects[flag['index']].name:
                self.gl.adv.update1( self.gl.objects[flag['index']])
            else:
                # self.gl.adv =copy.deepcopy( self.gl.objects[flag['index']])
                self.gl.adv = self.gl.objects[flag['index']]
        elif  front_close_vec_index != -1:
            self.gl.compete_time = 0
            pop_index.append(front_close_vec_index)
            # self.gl.adv = copy.deepcopy(self.gl.objects[front_close_vec_index])
            self.gl.adv = self.gl.objects[front_close_vec_index]
            # self.gl.objects.pop(front_close_vec_index)
            # self.gl.objects_set.remove(self.gl.objects[front_close_vec_index].id)
        else:
          #  print("ADV---ego front no compete vecs !!!")
            self.gl.adv = None
            self.gl.compete_time = 0
            return None
        # print("ADV---self.gl.adv.name:",self.gl.adv.name)
        # 从目标物维护列表中删除对抗车，以及较远的目标
        pop_index  =sorted(pop_index,reverse=True)
        # print("ADV---pop index:",pop_index)
        # print("ADV---self.gl.objects_set:",self.gl.objects_set)
        # for i in self.gl.objects:
        #     i.show()
        # print("ADV---88888888888888888888888888")
        self.gl.compete_time += 1
        for p in pop_index:
            # print("ADV---p:",p)
            # print("ADV---self.gl.objects[p].id:",self.gl.objects[p].name,self.gl.objects[p].id)
            self.gl.objects_set.remove(self.gl.objects[p].id)
            self.gl.objects.pop(p)
        if show_log:
            # print("ADV---len objs:",len(self.gl.objects))
            print("ADV---self.gl.adv:")
            self.gl.adv.show()
            print("ADV---self.gl.ego:")
            self.gl.ego.show()
        # 当前函数就是处理仿真世界当前帧数据并选取对抗车辆
        # return entry

    # 获取车辆距离
    def get_distance(self,ego):
        return math.sqrt( math.pow(ego.pos_x,2) + math.pow( ego.pos_y,2) )
    # 获取车辆速度
    def get_speed(self,vx,vy):
        return math.sqrt( math.pow(vx,2) + math.pow( vy,2) )
    def get_sqrt(self,x,y):
        return math.sqrt( math.pow(x,2) + math.pow( y,2) )
    ##########################################################
    ##########################################################
    ##########################################################
    # 获取阶梯速度索引， 用来计算预期加速度
    def get_speed_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.DEFAULT_TARGET_SPEEDS[0]) / (self.DEFAULT_TARGET_SPEEDS[-1] - self.DEFAULT_TARGET_SPEEDS[0])
        return np.int64(  np.clip(
            np.round(x * (self.DEFAULT_TARGET_SPEEDS.size - 1)), 0, self.DEFAULT_TARGET_SPEEDS.size - 1)  )
    # speed m/s

    def lmap(self, v: float, x, y) -> float:
        """Linear map of value v with range x to desired range y."""
        return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    def get_max_acc_new(self,speed):
        speed = self.lmap(speed, [5, 20], [0, 1])
        speed = self.lmap(speed, [0, 1], [-5, -3.5])
        max_acc = np.clip(speed, -5, -3.5)
        return max_acc
    def get_max_acc(self,speed):
        # 根据速度不同，法规所规定的最大加速度限制
        if speed < 5:
            return -5
        elif speed > 20:
            return -3.5
        else:
            return -speed*0.1 - 5.5

    # 左变道
    def lane_left_act(self):
        self.gl.scp.turn_left(self.gl.adv.name)

    # 保持不变
    def idle_act(self):
        # pass
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, self.gl.adv.id  , 1)
    # 右变道
    def lane_right_act(self):
        self.gl.scp.turn_right(self.gl.adv.name)

    # 加速指令
    def faster_act(self,var = 1):
        # 此时坐标系已转换为对抗车坐标系下
        # 对抗车绝对速度
        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)
        # 获取预期速度索引
        # 获取目标速度 target_speed
        speed_index = self.get_speed_index(adv_speed)
        if adv_speed < self.DEFAULT_TARGET_SPEEDS[ 1 ] and var == -1:
            target_speed = self.DEFAULT_TARGET_SPEEDS[ 0 ] + 1
        elif adv_speed > self.DEFAULT_TARGET_SPEEDS[ -1 ] and var == 1:
            target_speed = adv_speed + 3
        else:
            # 防止索引溢出
            if speed_index >= len(self.DEFAULT_TARGET_SPEEDS) - 1:
                print("ADV---adv speed > 30m/s, stop!!!")
                var = -1
            target_speed = self.DEFAULT_TARGET_SPEEDS[ speed_index + var ]
        # 预期加速度
        acceleration = target_speed  - adv_speed
        print("ADV---lon acceleration: ", acceleration)
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], acceleration, 0, self.gl.adv.id  , 1)

    def opt_conver_acc(self,acceleration):
        # print("ADV---self.C_time:",self.C_time)
        # print("ADV---self.C_time:",self.C)
        # print("ADV---acceleration:",acceleration)
        # 加速度正负反转状态，处理加速度指令由2 变为 -2 的突变情况
        if self.C_time >=0  and  self.C_time < 4:
            self.C  = acceleration
            acceleration = self.C_acc[self.C_time]
            self.C_time += 1
            if self.C_time >= 4:
                self.C_time  = -1
                self.C_acc.clear()

        elif ((self.C > 0 and acceleration < 0 ) or (self.C < 0 and acceleration > 0)) or abs(self.C -acceleration) > 2  :
            if self.C < acceleration:
                self.C_acc.append(self.C + abs(acceleration -self.C)*0.2)
                self.C_acc.append(self.C + abs(acceleration -self.C)*0.4)
                # self.C_acc.append(self.C + (acceleration -self.C)*0.5)
                self.C_acc.append(self.C + abs(acceleration -self.C)*0.6)
                self.C_acc.append(self.C + abs(acceleration -self.C)*0.8)
            elif self.C > acceleration:
                self.C_acc.append(self.C - abs(self.C -acceleration)*0.2)
                self.C_acc.append(self.C - abs(self.C -acceleration)*0.4)
                # self.C_acc.append(acceleration + (self.C -acceleration)*0.5)
                self.C_acc.append(self.C - abs(self.C -acceleration)*0.6)
                self.C_acc.append(self.C - abs(self.C -acceleration)*0.8)
            self.C_time  = 0
            acceleration = self.C_acc[self.C_time]
            self.C_time  +=1
        else:
            self.C = self.C  = acceleration

        # print("ADV---self.C_acc:",self.C_acc)
        return acceleration


    def slower_act(self):
        self.faster_act(-1)
    def left_1_act(self):
        self.gl.scp.Laneoffset(self.gl.adv.name, 1.2)
    def left_2_act(self,offset):
        self.gl.scp.Laneoffset(self.gl.adv.name, offset)
    def right_1_act(self):
        self.gl.scp.Laneoffset(self.gl.adv.name, -1.2)
    def right_2_act(self,offset):
        self.gl.scp.Laneoffset(self.gl.adv.name, -offset)
    def LCA(self):
        self.gl.scp.Laneoffset(self.gl.adv.name, 0)

    def LDWS(self):  # 车道偏离

        if abs(self.gl.adv.leftLaneId ) > 30 and self.action_marking in ['LEFT_2','LANE_LEFT']:
            self.action_marking = ''
            print("ADV---left 车道偏离")
            # self.right_1_act()
            self.LCA()

        if abs(self.gl.adv.rightLaneId ) > 30 and self.action_marking in ['RIGHT_2','LANE_RIGHT']:
            print("ADV---right 车道偏离")
            self.action_marking ='LEFT_1'
            self.LCA()
            # # self.left_1_act()
    def exec_lon_act(self):
        print("ADV--- real  contral", self.ctrl_signal)
        if self.ctrl_signal['lon'] == 'FASTER':
            self.faster_act()
        elif self.ctrl_signal['lon'] == 'SLOWER':
            self.slower_act()
        elif self.ctrl_signal['lon'] == 'IDLE':
            self.idle_act()

    # 横向执行函数
    def exec_lat_act(self):
        # self.ctrl_signal['lat'] = "RIGHT_2"
        # 车道偏移offset， 如果对抗车在安全距离以内L
        gain = 0
        lane_off_set = 1.4
        if self.dis < self.safe_dis :
            gain = random.uniform(0.2,0.6)
        lane_off_set += gain

        if self.ctrl_signal['lat']  == "LANE_LEFT"  :
            self.lane_left_act()
            return True
        elif self.ctrl_signal['lat'] == "LEFT_2" :
            self.left_2_act(lane_off_set)
            return True
        elif self.ctrl_signal['lat'] == "LANE_RIGHT" :
            self.lane_right_act()
            return True
        elif self.ctrl_signal['lat'] == "RIGHT_2":
            self.right_2_act(lane_off_set)
            return True
        else:
            self.gl.scp.auto(self.gl.adv.name)
            self.keep_time = -1
            self.keep_time_index = -1
            self.action_marking = ""
            return False

    def stop_lat_act(self,flag = True):
        # 执行完横向动作(BLANK和IDLE除外)之后，需要进入BLANK缓冲期
        if self.action_marking in ["BLANK","IDLE",""]:
            self.keep_time = -1
            self.keep_time_index = -1
            self.action_marking = ''
            if flag:
                self.gl.scp.overLaneoffset(self.gl.adv.name)
        else:
            if self.action_marking in ["LANE_RIGHT","LANE_LEFT"] and self.gl.adv.lane_id != self.gl.ego.lane_id:
                vx,vy = trans2angle(self.gl.adv.vx,self.gl.adv.vy,self.gl.adv.pos_h)
                go2dis = (abs(self.gl.ego.pos_x)/ ((self.gl.ego.vx + vx)  if (self.gl.ego.vx + vx)  > 0 else    0.1)    ) * vx + abs(self.gl.ego.pos_x)
                time1_fram_num = int( go2dis/(self.gl.ego.vx + vx)*20 )
                print("ADV--- time_fram_num:",time1_fram_num)
                self.keep_time = self.ACTIONS_DUR["BLANK"] + time1_fram_num if time1_fram_num < 220 else 220
            else:
                self.keep_time = self.ACTIONS_DUR["BLANK"] + 10
            self.keep_time_index = -1
            self.action_marking = "BLANK"
        # 执行纵向指令
        self.exec_lon_act()
    def check_rel_speed(self,lat_warn,ego_speed,adv_speed):
        # 将主车与对抗车的速度差不要太大，然后在开始对抗
        # 如果主车速度比对抗车快 并且 对抗车速度没有达到主车速度的0.7倍，对抗车应该先加速以满足对抗要求
        # print("lat_warn:", lat_warn)
        # ADV BACK
        front_dis = 0
        if self.model_type in [0,2,3]:
            front_dis= 0
        if self.gl.ego.pos_x < front_dis and abs(self.gl.ego.pos_x) > 3 * self.safe_dis:
            if  self.get_speed(self.gl.adv.vx, self.gl.adv.vy) < 5:
                self.ctrl_signal['lon'] = "FASTER"
            elif self.gl.ego.vx < 0 or abs(self.gl.ego.pos_x) > 5 * self.safe_dis :
                print("-----------对抗车速度过快，创造对抗条件------------")
                self.ctrl_signal['lon'] = "SLOWER"
                return True
        # ADV FRONT
        if self.gl.ego.pos_x > front_dis:
            if self.gl.ego.pos_x > front_dis/2:
                print("ADV--- 对抗车速度过慢，创造对抗条件")
                self.ctrl_signal['lon'] = "FASTER"
            if self.gl.ego.vx > 0:
                print("ADV--- 对抗车速度过慢，创造对抗条件")
                self.ctrl_signal['lon'] = "FASTER"

                return True

            if self.gl.ego.vx < 0 and abs(self.gl.ego.vx) > (5 if self.model_type == 0 else 3):
                self.ctrl_signal['lon'] = "SLOWER"
                return True

        # if abs(self.gl.ego.pos_x) > 30:
        #     if (self.gl.ego.vx >  0 and  ego_speed*0.5 > adv_speed):
        #         print("lat_warn:",lat_warn)
        #         print("-----------对抗车速度过慢，创造对抗条件------------")
        #         self.ctrl_signal['lon'] = "FASTER"
        #         return True
        #     # 如果主车速度比对抗车慢，并且 对抗车速度大于主车速度的1.3倍，对抗车应该减速以满足对抗要求
        #     elif self.gl.ego.vx < 0 and   ego_speed*1.3 < adv_speed:
        #         print("-----------对抗车速度过快，创造对抗条件------------")
        #         self.ctrl_signal['lon'] = "SLOWER"
        #         return True

    def check_lane_change(self):
        print("ADV---  self.safe_dis:",self.safe_dis)
        print("ADV---  dis:",self.dis)
        # if  abs(self.gl.ego.pos_x)  <= self.safe_dis and self.model_type != 0:
        #     self.ctrl_signal['lon'] = "FASTER"
        # if self.gl.ego.pos_x < -30:
        #     self.ctrl_signal['lon'] = "SLOWER"
        safe_dis = self.safe_dis
        ahw = (self.gl.front_vec_to_compete if self.gl.front_vec_to_compete else (
            self.gl.left_neib_front_vec_to_compete if self.gl.left_neib_front_vec_to_compete else self.gl.right_neib_front_vec_to_compete))
        # 有墙，且对抗车前方有障碍物且不足35m时，适当放宽安全距离限制
        if self.model_type == 0 and ahw and  ahw.pos_x < 35:
            safe_dis  = safe_dis * 0.4
            print("new safe dis:",safe_dis)
        ## 对抗车左右两侧是否可以便道
        left_lane = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
        right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
        # 对抗车前方有行使车辆，便道处理
        if (ahw  and not ahw.static and ahw.pos_x < 10 and self.gl.adv.lane_id ==  ahw.lane_id):

            if left_lane and  self.check_lane_change_action(self.gl.left_neib_front_vec_to_compete, self.gl.left_neib_bake_vec_to_compete):
                self.ctrl_signal['lat'] = "LANE_LEFT"
                return False
            elif right_lane and self.check_lane_change_action(self.gl.right_neib_front_vec_to_compete,self.gl.right_neib_bake_vec_to_compete):
                self.ctrl_signal['lat'] = "LANE_RIGHT"
                return False
        # 对抗车在主车前方时
        if self.gl.adv.direction_to_ego == 0:
            if self.ctrl_signal['lat'] == 'LANE_RIGHT' and self.gl.right_neib_bake_vec_to_compete and abs(self.gl.right_neib_bake_vec_to_compete.pos_x) < abs(self.gl.ego.pos_x) :
                right_lane = 0
                return True
            if self.ctrl_signal['lat'] == 'LANE_LEFT' and self.gl.left_neib_bake_vec_to_compete and  abs(self.gl.left_neib_bake_vec_to_compete.pos_x) < abs(self.gl.ego.pos_x):
                left_lane = 0
                return True
            # lane_off_set -= gain
            print("ADV---left_lane and right_lane:",left_lane,right_lane)
        #  主车在对抗车右侧，禁止左方向移动
        if self.ctrl_signal['lat'] == 'LANE_LEFT' and self.gl.ego.pos_y < -2:
            left_lane = 0
            return True
        #  主车在对抗车左侧，禁止右方向移动
        if self.ctrl_signal['lat'] == 'LANE_RIGHT' and  self.gl.ego.pos_y > 2:
            right_lane = 0
            return True
        # 截至路处理
        adv_current_lon, _ = self.gl.ego.lane.local_coordinates(np.array([self.gl.adv.pos_x, self.gl.adv.pos_y]))
        if self.ctrl_signal['lat'] == 'LANE_LEFT' and left_lane and\
                self.map.openDriveXml.getRoad(self.gl.adv.roadId).lanes.lane_sections[0].lane_width_dict[
                    self.gl.adv.leftLaneId][int(adv_current_lon)] < self.gl.norm_lane_width*0.95:
            return True
        if self.ctrl_signal['lat'] == 'LANE_RIGHT' and right_lane and\
                self.map.openDriveXml.getRoad(self.gl.adv.roadId).lanes.lane_sections[0].lane_width_dict[
                    self.gl.adv.rightLaneId][int(adv_current_lon)] < self.gl.norm_lane_width*0.95:
            return True
        # 在小于安全距离，且对抗车在相邻车道时，或者，对抗车在主车后方且与主车不同车道时，不能换道
        return (self.gl.ego.pos_x > -safe_dis and abs(self.gl.adv.lane_id -  self.gl.ego.lane_id) == 1 ) or (self.gl.ego.pos_x > 0  and  self.gl.adv.lane_id !=  self.gl.ego.lane_id)

    def contrl_adv(self,lat_warn):
        # get adv and ego absolute  speed
        adv_vx, adv_vy = trans2angle(self.gl.adv.vx, self.gl.adv.vy, self.gl.adv.pos_h)
        adv_speed = self.gl.adv.get_sqrt(adv_vx, adv_vy)
        ego_speed = self.get_speed( adv_vx + self.gl.ego.vx, adv_vy + self.gl.ego.vy )
        print("ADV---ego_speed_vx:", ego_speed)
        print("ADV---adv_speed:", adv_speed)
        #对抗目标若是行人，走此分支
        # if self.gl.adv.obj_type == 5:
        #     if self.ctrl_signal['lon'] == 'FASTER':
        #         if adv_speed > 2:
        #             adv_speed = 2
        #         else:
        #             adv_speed += 1
        #         self.gl.scp.dacc(actor=self.gl.adv.name, target_speed = adv_speed , type=5  )
        #     # self.gl.scp.dacc(self,actor, target_speed = 20,type = None):
        #     elif self.ctrl_signal['lon'] == 'SLOWER':
        #         if adv_speed < 0.3:
        #             adv_speed = 0.3
        #         else:
        #             adv_speed  -= 0.5
        #         self.gl.scp.dacc(actor=self.gl.adv.name, target_speed = adv_speed , type=5  )
        #     elif self.ctrl_signal['lon'] == 'IDLE':
        #         pass
        #     return
            # self.stop_lat_act()
        # 慢跟车处理逻辑
        slow_check = self.slow_following_check(lat_warn)
        # slow_check  = None
        if slow_check is not None:
            print("slow follwing !!!  slow_check:",slow_check)
            self.ctrl_signal['lat'] = slow_check
            if slow_check != 'IDLE':
                self.ctrl_signal['lon'] = 'FASTER'

        # 检查主车与对抗车速度差距是否过大
        self.check_rel_speed(lat_warn,ego_speed,adv_speed)
        # 横向控制指令  self.keep_time 动作预留时间，若为负，则更新，进入横向控制阶段
        # 横向控制指令如果有碰撞风险，或 对抗车在主车后面且在不同车道 则取消横向控制指令下发
        # print("lat_warn:",lat_warn)
        if self.check_lane_change():
            # 如果此时正在进行变道动作期间，则动作无法撤回,即变道动作已经执行，横向警告拦截失败
            if self.action_marking in ["LANE_LEFT", "LANE_RIGHT"]:
                print("ADV---lane change action can not stop!!!")
            else:
                print("ADV---stop lat act!!!")
                self.stop_lat_act()
                return  -1

        print(">>>>>>>>>>>>>>>>>>>")
        print("keep_time:",self.keep_time)
        print("keep_time_index:", self.keep_time_index)
        print("action_marking:", self.action_marking)
        print(">>>>>>>>>>>>>>>>>>>")
        # 有墙模式下，偏移动作优先级低于变道动作，优先执行变道指令
        if self.model_type == 0 and (self.ctrl_signal['lat'] in ['LANE_LEFT','LANE_RIGHT'] ) and(self.action_marking not in ['LANE_LEFT','LANE_RIGHT','']): 
            if self.exec_lat_act():
                self.keep_time = self.ACTIONS_DUR[self.ctrl_signal['lat']]
                self.action_marking = self.ctrl_signal['lat']
                self.keep_time_index  = 0

        # 是否执行横向对抗， keep_time == -1 无任何正在执行横向动作， 此时可以进入横向对抗动作
        if self.keep_time <= 0 and  self.ctrl_signal['lat'] != "IDLE" :
 
            # if (self.ctrl_signal['lat'] == "LANE_LEFT" and self.check_lane_change_action(self.gl.left_neib_front_vec_to_compete, self.gl.left_neib_bake_vec_to_compete) ) or \
            #         (self.ctrl_signal['lat'] == "LANE_RIGHT" and self.check_lane_change_action(self.gl.right_neib_front_vec_to_compete, self.gl.right_neib_bake_vec_to_compete) ):
            if self.exec_lat_act():
                self.keep_time = self.ACTIONS_DUR[self.ctrl_signal['lat']]
                self.action_marking = self.ctrl_signal['lat']
                self.keep_time_index  = 0
            # if self.ctrl_signal['lat'] in ['LANE_RIGHT','LANE_LEFT']:
            #     # if (self.ctrl_signal['lat'] == "LANE_LEFT" and self.check_lane_change_action(self.gl.left_neib_front_vec_to_compete, self.gl.left_neib_bake_vec_to_compete) ) or \
            #     #         (self.ctrl_signal['lat'] == "LANE_RIGHT" and self.check_lane_change_action(self.gl.right_neib_front_vec_to_compete, self.gl.right_neib_bake_vec_to_compete) ):
            #         self.exec_lat_act()
            #         self.keep_time = self.ACTIONS_DUR[self.ctrl_signal['lat']]
            #         self.action_marking = self.ctrl_signal['lat']
            # else:
            #     self.exec_lat_act()

        elif self.keep_time_index >= self.keep_time:
            # 如果是变道指令，并且预留时间已经用完，但是变道还未完成，则再增加5帧预留时间，预留时间上限为120帧,120帧后强制清零
            if self.action_marking in ['LANE_RIGHT','LANE_LEFT'] and self.gl.adv_hdg_num <= 30 and self.keep_time < 120 :
                self.keep_time += 5
            else:
                self.stop_lat_act()
                return -1
        # 查看对抗车车头是否与车道朝向同方向
        if self.action_marking in ['LANE_RIGHT', 'LANE_LEFT'] and self.keep_time_index >= int(self.keep_time/2):
            if abs(self.gl.adv.hdg) <= 0.02:
                self.gl.adv_hdg_num += 1
            else:
                self.gl.adv_hdg_num = 0

        if self.action_marking:
            self.keep_time_index += 1
        print(">>>>>>>>>>>>>>>>>>>:", self.action_marking)
        # if self.action_marking in ['LANE_RIGHT','LANE_LEFT'] and self.keep_time_index < int(self.keep_time/3):
        #     self.ctrl_signal['lon'] = "IDLE"
        # 纵向控制指令
        # 执行纵向指令
        self.exec_lon_act()

    def slow_following_check(self, lat_warn):
        result = None
        if self.gl.ego.lane_id != self.gl.adv.lane_id:
            return result
        # 慢跟车处理逻辑
        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)

        # 如果对抗车速度小于3m/s，则抑制掉横向对抗动作，此处不影响慢车换道逻辑，result会在满足慢车换道逻辑后重新刷新
        if adv_speed <= 3:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ADV speed < 3!!!!!!!!!!!!")
            result =  'IDLE'
        # 当速度小于1.5时，启动换道触发逻辑
        if adv_speed < 1.5 or abs(self.gl.ego.vx) < 2 and self.gl.bake_vec_to_compete is not None:
            if self.gl.adv_pre_speed < 1.5:
                # 如果主车当前帧和上一帧都是慢速，slow_dur加一
                self.gl.slow_dur += 1  # 在GLOBAL里添加属性
                # 如果慢速跟车持续300 帧以上，则执行换道指令
                print("slow_dur: ", self.gl.slow_dur)
                if self.gl.slow_dur > 300 and not lat_warn:
                    # 对抗车左侧车道可通行，优先左边到
                    if abs(self.gl.adv.leftLaneId) < 30:
                        self.lane_left_act()
                        result = "LANE_LEFT"
                    elif abs(self.gl.adv.rightLaneId) < 30:
                        result = "LANE_RIGHT"
                    else:
                        print("no lane to turn!!!")
                        pass
                    # 如果计时大于300，也需要清零
                    self.gl.slow_dur = 0
        # 当速度大于3时，一定要清零计时
        elif adv_speed >= 3:
            self.gl.slow_dur = 0
        self.gl.adv_pre_speed = adv_speed
        return result


    # 处理一帧数据
    def vtd_exec_atomic(self):
        self.gl.clear_fram_data()
        self.parseRDBMessage()
    def get_sqrt(self,x,y):
        return math.sqrt(math.pow(x,2) + math.pow(y,2))

    def run(self,action):
        self.vtd_exec_atomic(action)
    # return array 1* 36
    def space(self) -> spaces.Space:
        state_dim = 11
        if self.model_type == 0:
            state_dim = self.args.model_config_list['model_wall'].state_dim
        elif  self.model_type == 1:
            state_dim = self.args.model_config_list['model_lon'].state_dim
        elif self.model_type == 2:
            state_dim = self.args.model_config_list['model_no_wall'].state_dim
        elif self.model_type == 3:
            state_dim = self.args.model_config_list['model_dynamic_wall'].state_dim
        return spaces.Box(shape=(1, state_dim), low=-np.inf, high=np.inf,
                          dtype=np.float32)

    # 控制指令执行函数
    def contrl(self,lat_warn):
        self.contrl_adv(lat_warn)
        # 如果没有任何纵向加速信息，则保持
        if self.lib.get_msg_num()  ==  0 and self.gl.adv is not None:
            self.lib.addPkg(   self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, self.gl.adv.id, 1)


    # 碰撞警告，且会将坐标系转换成对抗车辆坐标系
    def collision_warning(self):
        # threshold
        # 碰撞预警
        ttc = -1
        ttc_threshold = 5
        safe_dis_threshold = 5
        safe_distance2ego = 0
        lon_warn = False
        lat_warn = False
        if self.gl.front_vec_to_compete is not None and  self.gl.front_vec_to_compete.static and  (self.gl.front_vec_to_compete.pos_x < 10 and self.model_type in [0,3] and self.gl.ego.pos_x > -6 and self.gl.front_vec_to_compete.lane_id == self.gl.adv.lane_id):
            lon_warn = True
        #print("ADV---safe_distance2ego: ", safe_distance2ego)
        elif self.gl.front_vec_to_compete is not None and not  self.gl.front_vec_to_compete.static:
            # self.gl.front_vec_to_compete.show()   -5 减去主车与对抗车车身长度
            # print("ADV---self.gl.front_vec_to_compete.vx:",self.gl.front_vec_to_compete.vx)
            if  self.gl.front_vec_to_compete.pos_x < 5:
                lon_warn = True
            elif self.gl.front_vec_to_compete.vx < 2:
                ttc =(self.get_sqrt(self.gl.front_vec_to_compete.pos_x , self.gl.front_vec_to_compete.pos_y))  /  self.get_speed(self.gl.front_vec_to_compete.vx, self.gl.front_vec_to_compete.vy)
                safe_distance2ego = self.get_sqrt(self.gl.front_vec_to_compete.pos_x , self.gl.front_vec_to_compete.pos_y) - 5.5
                print("ADV---TTC:", ttc, safe_distance2ego)
                if ttc < ttc_threshold and safe_distance2ego < safe_dis_threshold:
                    print("ADV---collision warning with front vec!!!  ttc:",ttc)
                    lon_warn = True

        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)
        safe_dis = (self.gl.adv.vx + self.gl.ego.vx)*0.1 + 4 + math.pow(self.gl.ego.vx,2) / (2*(-self.get_max_acc_new(adv_speed) - 0.03* ((self.gl.adv.vx + self.gl.ego.vx)/10)   ) )


        if self.ctrl_signal['lat']  == "LANE_LEFT" and  (self.gl.left_neib_front_vec_to_compete or  self.gl.left_neib_bake_vec_to_compete):
            lat_request = self.lat_check.mobil(self.gl.adv, self.gl.left_neib_front_vec_to_compete, self.gl.left_neib_bake_vec_to_compete,
                None, None, direction='LEFT')
        elif self.ctrl_signal['lat']  == "LANE_RIGHT" and (self.gl.right_neib_front_vec_to_compete or self.gl.right_neib_bake_vec_to_compete):
            lat_request = self.lat_check.mobil(self.gl.adv, self.gl.right_neib_front_vec_to_compete, self.gl.right_neib_bake_vec_to_compete,
                None, None, direction='RIGHT')
        else:
            lat_request = True
        if lat_request == True :
            lat_warn = False
        else:
            lat_warn = True
        # for i in other_vecs:
        #     i.show()
        return lon_warn, lat_warn
    # 急刹指令
    def stop(self,actor_id):
        self.lib.clear()
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -4, 0, actor_id  , 1)
        print("ADV--->>>>>>>>>>>>>>>>>>>>ctrl>>>>>>>>>>>>>>>>>>>>: stop")
    # 控制指令发送
    def sendctrlmsg(self):
        self.lib.sendTrigger(self.sClient, self.gl.fram_data['simTime'], self.gl.fram_data['simFrame'],0  )
        self.lib.clear()
    # 自动驾驶
    def autopilot(self,actor_id):
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, actor_id  , 0)
        #
    # 碰撞检测
    def collision_check(self,vec0,vec1):

        v0_p = []
        v0_p.append([ vec0.off_x  - (vec0.l*0.5)   ,-vec0.w*0.5 ] )
        v0_p.append([ vec0.off_x  - (vec0.l*0.5)   , vec0.w*0.5,  ])
        v0_p.append([ vec0.off_x  + (vec0.l*0.5)   , vec0.w*0.5,   ])
        v0_p.append([ vec0.off_x  + (vec0.l*0.5)   ,-vec0.w*0.5,  ])
        # print("ADV---v0_p:",v0_p)
        v0_p = np.array(v0_p)

        # print("ADV---vec1.pos_h",vec1.pos_h)
        center_x  = vec1.pos_x +  vec1.off_x * math.cos(vec1.pos_h)
        center_y  = vec1.pos_y +  vec1.off_x * math.sin(vec1.pos_h)
                                        # pi /  2
        center = [center_x,center_y]
        v1_p = []
        # v1_p.append( rotate_operate(center_x - vec1.l*0.5 ,  center_y - vec1.w * 0.5, vec1.pos_h)           )
        # v1_p.append(  rotate_operate(center_x - vec1.l*0.5 ,  center_y + vec1.w * 0.5,vec1.pos_h)           )
        # v1_p.append(  rotate_operate(center_x + vec1.l*0.5 ,  center_y + vec1.w * 0.5,vec1.pos_h)           )
        # v1_p.append(  rotate_operate(center_x + vec1.l*0.5 ,  center_y - vec1.w * 0.5,vec1.pos_h)           )

        v1_p.append( [   rotate_operate( - vec1.l*0.5 ,   - vec1.w * 0.5, vec1.pos_h)[i] + center[i] for i in range(len(center))  ]           )
        v1_p.append(  [  rotate_operate( - vec1.l*0.5 ,    vec1.w * 0.5,vec1.pos_h)[i] + center[i]   for i in range(len(center))  ]           )
        v1_p.append( [rotate_operate(  vec1.l*0.5 ,    vec1.w * 0.5,vec1.pos_h)[i] + center[i]   for i in range(len(center))  ]          )
        v1_p.append(  [rotate_operate(  vec1.l*0.5 ,   - vec1.w * 0.5,vec1.pos_h)[i] + center[i]   for i in range(len(center))   ]         )


        # print("ADV---v1_p:",v1_p)
        # v1_p  = [ [-0.9999999196153099, -3.0000000267948956],[-1.0000000803846893, 2.9999999732051026], [0.9999999196153099, 3.0000000267948956], [1.0000000803846893, -2.9999999732051026]]
        # print("ADV---v1_p:",v1_p)
        # for i in v1_p:
        #     temp = list(map(lambda x,y:x + y, vp_p0,i))
        #     temp = list(map(lambda x,y:x + y, vp_p0,i))
        #     i[0] = temp[0]
        #     i[1] = temp[1]
        v1_p = np.array(v1_p)
        # print("ADV---v0_p:",v0_p)
        # print("ADV---v1_p:",v1_p)
        poly1 = Polygon(v0_p).convex_hull
        poly2 = Polygon(v1_p).convex_hull
        # print("ADV---poly:",poly1,poly2)
        union_poly = np.concatenate((v0_p,v1_p))   #合并两个box坐标，变为8*2
        # print("ADV---union_poly:",union_poly)
        if not poly1.intersects(poly2): #如果两四边形不相交
            # print("ADV---如果两四边形不相交")
            return False

        else:
            # inter_area = poly1.intersection(poly2).area   #相交面积
            print("ADV--->>>>>>>>>>已碰撞!!!>>>>>>>>>>>>>>>>")
            return True

        #     # print("ADV---inter_area",inter_area)
        #     #union_area = poly1.area + poly2.area - inter_area
        #     union_area = MultiPoint(union_poly).convex_hull.area
        #     # print("ADV---union_area:",union_area)
        #     if union_area == 0:
        #         iou= 0
        #     print("ADV---inter_area/union_area:",inter_area, union_area)
        #     iou=float(inter_area) / union_area
        #     print("ADV---iou:",iou)
        #     # gap    fix
        #     if iou > 0.013:
        #         return True
        # return False
    def get_time(self,obj):
        time = -1
        s = self.get_sqrt(self.prepare.confrontation_position[0] - obj.pos_x, self.prepare.confrontation_position[1] - obj.pos_y)
        v = self.get_sqrt(obj.vx,  obj.vy)
        # print("ADV---s:",s)
        time = 99
        if v > 0:
            time = s / v
        return time,s,v
    # npc 车辆控制函数
    def vtd_func(self):
        # 如果相遇点不存在，则直接返回  即 confrontation1 trigger
        if self.prepare.confrontation_position[0] == -1:
            return
        # 计算与主车到达相遇点时间，如果大于主车，则加速，如果小于主车，则减速
        ego_time,ego_s, ego_v = self.get_time(self.gl.ego)

        for i in self.gl.objects:
            # if i.name != 'npc':
            #     continue
            if ego_v < 1:
                self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 1, 0, i.id, 1)
                continue
            time,s,v = self.get_time(i)
            if i.direction_to_ego == 0:
                time +=5
            elif i.direction_to_ego == 1:
                time -=2
            else:
                time += 0.5
            if time < 0:
                continue
            print("ADV---time, ego_time:",time, ego_time)
            _ , in_juction = self.chek_in_juction( i.roadId, 0,i.lane_id)
            if  abs(s) < 5 and v > 0 :
                self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -4, 0, i.id, 1)

            elif self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == 'STOP':
                print("ADV---decrease acc!!!",i.id)
                if  self.get_sqrt( i.vx,  i.vy) > 0:
                    self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -3, 0, i.id, 1)
                else :
                    self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, i.id, 1)
            # if self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == '':
            elif  time > ego_time:
                # print("ADV---~~~~~~~~~~~~~~~~~~~~~~~~~~~i.name:",i.name)
                self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 3, 0, i.id, 1)
            elif time <= ego_time:
                print("ADV---decrease acc!!!")
                self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -5, 0, i.id, 1)

    def trans_world_cood(self):
        # 转换为全局坐标系
        if self.gl.adv is not None:
            self.gl.adv.to_dict(self.gl.ego, True, True,True,name=1)
        index = 2
        for i in self.gl.objects:
            i.trans_cood1(self.gl.ego, True,True,True)
            index+=1
    def trans_to_world(self,base_obj,trans_obj,position_flag = False,velocity_flag = False,acc_flag = False,rotate = -1):
        theta = rotate * base_obj.pos_h
        result = dict()
        # print("theta:",theta)

        # if position_flag:
        #     obj.pos_x =  pos_x*math.cos(theta) + pos_y*math.sin(theta)
        #     obj.pos_y = -(pos_y*math.cos(theta) - pos_x*math.sin(theta))
        if position_flag:
            pos_x = trans_obj.pos_x
            pos_y = trans_obj.pos_y
            new_pos_x = pos_x * math.cos(theta) + pos_y * math.sin(theta) + base_obj.pos_x
            new_pos_y = pos_y * math.cos(theta) - pos_x * math.sin(theta) + base_obj.pos_y
            result["pos_x"] = new_pos_x
            result["pos_y"] = new_pos_y
        if velocity_flag:
            vx = trans_obj.vx
            vy = trans_obj.vy
            new_vx =  vx*math.cos(theta) + vy*math.sin(theta) + base_obj.vx
            new_vy =  vy*math.cos(theta) - vx*math.sin(theta)   + base_obj.vy
            result["vx"] = new_vx
            result["vy"] = new_vy
        if acc_flag:
            acc_x = trans_obj.acc_x
            acc_y = trans_obj.acc_y
            # print("org acc:",self.acc_x,self.acc_y)
            new_acc_x =  acc_x*math.cos(theta) + acc_y*math.sin(theta) + base_obj.acc_x
            new_acc_y =  acc_y*math.cos(theta) - acc_x*math.sin(theta)  + base_obj.acc_y
            result["accx"] = new_acc_x
            result["accy"] = new_acc_y
        return result


    def compute_ttc(self):
        if self.gl.front_vec_to_compete.name.find("static") != -1:
            result = self.trans_to_world(self.gl.adv,self.gl.front_vec_to_compete,position_flag = True)
            rel_dis = self.gl.front_vec_to_compete.lane.local_coordinates(np.array([result["pos_x"] ,
                                                                          result["pos_y"] ]) )[0] - \
                      self.gl.adv.lane.local_coordinates(np.array([self.gl.adv.pos_x, self.gl.adv.pos_y]))[0]
        else:
            rel_dis = self.get_distance(self.gl.front_vec_to_compete)
        rel_speed = -1
        print(" self.gl.front_vec_to_compete.vx:", self.gl.front_vec_to_compete.vx)
        if self.gl.front_vec_to_compete.vx <= 0:
            rel_speed = -self.gl.front_vec_to_compete.vx
            ttc = (rel_dis + 10) / rel_speed if np.fabs(rel_speed ) > 1e-10 else 50
        else:
            ttc = 50
        print(">>>>>>>>>>>>>>>>>>>>rel dis:",rel_dis  ,"rel speed:",rel_speed ,   "ttc:",ttc)
        return ttc
    def LRlane_change(self):

        left_lane = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
        right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
        if self.gl.ego.pos_y < -2:
            left_lane = 0
        #  主车在对抗车左侧，禁止右方向移动
        if self.gl.ego.pos_y > 2:
            right_lane = 0
        if left_lane and not  self.check_lane_change_action(self.gl.left_neib_front_vec_to_compete,
                                                       self.gl.left_neib_bake_vec_to_compete):
            left_lane = 0
        if right_lane and not self.check_lane_change_action(self.gl.right_neib_front_vec_to_compete,
                                                          self.gl.right_neib_bake_vec_to_compete):
            right_lane = 0
        return left_lane,right_lane



    # 获取强化学习模型输入状态
    def get_dqn_state_new(self):
        # 对抗车车辆坐标系，右手坐标系
        # 此时 对抗车应该是全局坐标系下绝对值
        # 其他所有车辆均为对抗车车辆坐标系下相对坐标值
        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)
        if self.gl.ego.pos_x  < 0:
            self.gl.ego.pos_x -= self.gl.dangerous
        vx,vy = trans2angle(self.gl.adv.vx, self.gl.adv.vy, self.gl.adv.pos_h)
        self.safe_dis = (vx + self.gl.ego.vx)*0.1 + 6 + (0 if self.gl.ego.pos_x <0 and  self.gl.ego.vx <0  else math.pow(self.gl.ego.vx,2) / (2*(-self.get_max_acc_new(adv_speed) - 0.03* ((vx + self.gl.ego.vx)/10)   ) )   )
        # self.safe_dis *= self.gl.dangerous
        self.dis = self.get_distance(self.gl.ego)
        left_lane,right_lane  = self.LRlane_change()
        # adv_current_lon, _ = self.gl.ego.lane.local_coordinates(np.array([self.gl.adv.pos_x, self.gl.adv.pos_y]))
        # # print("+++++++++++++++++++++++++++++++++++++++:",self.gl.adv.lane.length, adv_current_lon)
        # distance_ratio = (30 - (self.gl.adv.lane.length - adv_current_lon)) / self.gl.adv.lane.length
        distance_ratio = 1
        # 主车相对于对抗车的  相对位置x，y，相对速度 vx  vy  ，  对抗车的绝对速度      主车相对于对抗车的相对航向角，
        # 左变道，右变道，distance_ratio, safe distance, ttc
        # print("++++++++++++++++++++obs ego.pos_xs/afe_dis:",-self.gl.ego.pos_x, self.safe_dis)
        obs = [
               self.gl.ego.pos_x, self.gl.ego.pos_y, self.gl.ego.vx, self.gl.ego.vy,\
               self.get_speed(self.gl.adv.vx,self.gl.adv.vy),\
               self.gl.ego.pos_h - 2 * np.pi if self.gl.ego.pos_h > np.pi else self.gl.ego.pos_h,\
               left_lane,  right_lane,  distance_ratio, -self.gl.ego.pos_x - (self.safe_dis),\
               ]
        # ttc_adv_with_dynamic_obstacle = 1.0
        ehw = self.ego_front_has_wall()
        ahw = (self.gl.front_vec_to_compete if self.gl.front_vec_to_compete else (
            self.gl.left_neib_front_vec_to_compete if self.gl.left_neib_front_vec_to_compete else self.gl.right_neib_front_vec_to_compete))
        # 如果对抗车在主车左右两侧 且主车前方有车，进入纵向模型分支
        if self.gl.adv.direction_to_ego in [1, 2, 4, 5] and ehw:
            self.model_type = 1
            obs[0] -= (random.randint(1,self.gl.lon_offset) if obs[0] < 0 else 0)
            obs[9] =  -obs[0] - (self.safe_dis)
            obs.append(1.0)
        # 如果主车前方无车，且对抗车前方，左前方，右前方有车且距离小于wall_far， 进入墙分支
        elif not ehw and  ahw and (ahw.pos_x < self.wall_far):
            offset = 5
            obs += [ahw.pos_x - offset, ahw.pos_y, ahw.vx, ahw.vy, ahw.pos_h]
            # 静态墙
            if ahw.static:
                print("ADV--- find static front_vec_to_compete!!!---", ahw.name)
                self.model_type = 0
                # rel_dis = self.get_distance(ahw) -  5
                rel_dis = ahw.pos_x - offset
                if ahw.pos_x < 0:
                    rel_dis = -rel_dis
                dis_ratio_with_dynamic_obstacle = 1.0
                if rel_dis > 80:
                    dis_ratio_with_dynamic_obstacle = 1.0
                elif rel_dis > 25:
                    dis_ratio_with_dynamic_obstacle = (rel_dis - 25) / 55
                elif rel_dis > 0:
                    dis_ratio_with_dynamic_obstacle = (rel_dis - 25) / 25
                else:
                    dis_ratio_with_dynamic_obstacle = -1.0
                obs.append(dis_ratio_with_dynamic_obstacle)
            # 动态墙
            else:
                # dynamic
                print("ADV--- find dynamic front_vec_to_compete!!!---",ahw.name)
                self.model_type = 3
                speed_diff = max(self.get_speed(vx,vy) - self.get_speed(ahw.vx + vx, ahw.vy + vy),15)
                dis_ratio_with_dynamic_obstacle = 1.0
                if ahw.lane_id == self.gl.adv.lane_id:
                    rel_dis = ahw.pos_x - 5
                    if rel_dis > 80:
                        dis_ratio_with_dynamic_obstacle = 1.0
                    elif  rel_dis > speed_diff:
                        dis_ratio_with_dynamic_obstacle = (rel_dis - speed_diff) / (80 - speed_diff)
                    elif rel_dis > 0:
                        dis_ratio_with_dynamic_obstacle = (rel_dis - speed_diff) / speed_diff
                    else:
                        dis_ratio_with_dynamic_obstacle = -1.0
                obs.append(dis_ratio_with_dynamic_obstacle)

        # 其他情况全部按照无强模型处理
        else:
            self.model_type = 2
            obs.append(1.0)
        print("ADV---observe state:",obs)
        obs = np.array(obs).astype(self.space().dtype)
        obs = torch.tensor(obs).unsqueeze(0)
        return obs
    def ego_front_has_wall(self):
        dis = self.wall_far + 30
        for i in self.gl.objects:
            if i.direction_to_ego == 0:
                # tmp = i.pos_x - self.gl.ego.pos_x
                tmp = i.pos_x
                if dis > tmp:
                    dis = tmp
        return True if ( dis < self.wall_far + 30) else False
    def get_action_key(self,action,name):
        action = action[0].numpy()
        print("obs tensor:",action)
        action = np.argmax(action, keepdims=True)[0]
        action = self.model_manager.model_config_list[name].actions[action]
        # action = self.model_manager.model.model_config_list[ name ]["actions"]
        # action = self.args.actions_1[action]
        return action
    def sample_actions_new(self,state):
        with torch.no_grad():
            if self.model_type == 0:
                # action_map = {"RIGHT_2":"LANE_RIGHT","LEFT_2":"LANE_LEFT","LANE_RIGHT":"LANE_RIGHT","LANE_LEFT":"LANE_LEFT"}
                print("ADV-- wall_model_net")
                if self.model_manager.model_config_list["model_wall"].depart:
                    print("ADV--- depart wall_model_net")
                    pass
                else:

                    # action = self.model_manager.model.wall_value_net( state )
                    action = self.model_manager.model.model_list["model_wall"](state)
                    action = self.get_action_key(action,"model_wall")
                if action in ["FASTER", "SLOWER", "IDLE"]:
                    return {'lon': action, 'lat': "IDLE"}
                else:
                    return {'lon': "IDLE", 'lat': action}
                # if action in ["FASTER", "SLOWER", "IDLE"]:
                #     return {'lon': action, 'lat': "IDLE"}
                # else:
                #     if self.gl.front_vec_to_compete and self.gl.front_vec_to_compete.pos_x < 60 + (self.gl.ego.vx + trans2angle(self.gl.adv.vx,self.gl.adv.vy,self.gl.adv.pos_h)[0] ) *5:
                #         lon = "FASTER" if self.action_marking not in ["LANE_LEFT","LANE_RIGHT"]   or self.gl.ego.pos_x  > 0 else "IDLE"
                #         return {'lon': lon, 'lat': action_map[action]}
                #     else:
                #         return {'lon': "IDLE", 'lat': action}

            # lon model
            elif self.model_type == 1:

                if self.model_manager.model_config_list["model_lon"].depart:
                    print("ADV--- depart lon_model_net")
                    pass
                else:
                    print("ADV--- lon_model_net")
                    action = self.model_manager.model.model_list["model_lon"](state)
                    action = self.get_action_key(action,"model_lon")
                if action in ["FASTER", "SLOWER", "IDLE"]:
                    return {'lon': action, 'lat': "IDLE"}
                return {'lon': "IDLE", 'lat': "IDLE"}
            elif self.model_type == 2:
                if self.model_manager.model_config_list["model_no_wall"].depart:
                    print("ADV--- depart no_wall_model_net")
                    pass
                else:
                    print("ADV--- no_wall_model_net")
                    # action = self.model_manager.model.wall_value_net( state )
                    action = self.model_manager.model.model_list["model_no_wall"](state)
                    action = self.get_action_key(action,"model_no_wall")
                if action in ["FASTER", "SLOWER", "IDLE"]:
                    return {'lon': action, 'lat': "IDLE"}
                else:
                    return {'lon': "IDLE", 'lat': action}
            elif self.model_type == 3:
                # action_map = {"RIGHT_2":"LANE_RIGHT","LEFT_2":"LANE_LEFT","LANE_RIGHT":"LANE_RIGHT","LANE_LEFT":"LANE_LEFT"}
                print("ADV-- dynamic_wall_model_net")
                if self.model_manager.model_config_list["model_wall"].depart:
                    print("ADV--- depart wall_model_net")
                    pass
                else:
                    # action = self.model_manager.model.wall_value_net( state )
                    action = self.model_manager.model.model_list["model_dynamic_wall"](state)
                    action = self.get_action_key(action,"model_dynamic_wall")
                if action in ["FASTER", "SLOWER", "IDLE"]:
                    return {'lon': action, 'lat': "IDLE"}
                else:
                    return {'lon': "IDLE", 'lat': action}
    def trans_world_to_local(self):
        for i in self.gl.objects:
            # (self, base_obj, position_flag = False,velocity_flag = False,acc_flag = False ,rotate = -1):
            i.trans_cood2(self.gl.adv, True, True, True, rotate=1)
        self.gl.ego.trans_cood2(self.gl.adv, True, True, True, rotate=1)
    def get_adv_around_state(self):

        other_vecs = []
        other_vecs += self.gl.objects
        other_vecs.append(self.gl.ego)
        if self.gl.adv is not None:
            other_vecs = sorted(other_vecs, key=lambda x: math.sqrt(x.pos_x ** 2 + x.pos_y ** 2))
            num = 0
            threshold = 1.6
            # 获取对抗车前后左右车辆
            for obj in other_vecs:
                # if self.gl.front_vec_to_compete is None and obj.pos_x >= 0 and abs(obj.pos_y) <= threshold:
                if self.gl.front_vec_to_compete is None and obj.pos_x >= 0 and abs(obj.pos_y) <= threshold:
                    self.gl.front_vec_to_compete = obj
                    num += 1
                if  self.gl.left_neib_front_vec_to_compete is None and obj.pos_x >= 0 and obj.pos_y > threshold and abs(abs(obj.lane_id) - abs(self.gl.adv.lane_id) ) == 1:
                    self.gl.left_neib_front_vec_to_compete = obj
                    num += 1
                if self.gl.right_neib_front_vec_to_compete is None and obj.pos_x >= 0 and obj.pos_y < -threshold and abs(abs(obj.lane_id) - abs(self.gl.adv.lane_id) ) == 1:
                    self.gl.right_neib_front_vec_to_compete = obj
                    num += 1
                if self.gl.bake_vec_to_compete is None and obj.pos_x < 0 and abs(obj.pos_y) <= threshold:
                    self.gl.bake_vec_to_compete = obj
                    num += 1
                if self.gl.left_neib_bake_vec_to_compete is None and obj.pos_x < 0 and obj.pos_y > threshold  and abs(abs(obj.lane_id) - abs(self.gl.adv.lane_id) ) ==1 :
                    self.gl.left_neib_bake_vec_to_compete = obj
                    num += 1
                if self.gl.right_neib_bake_vec_to_compete is None and obj.pos_x < 0 and obj.pos_y < -threshold and abs(abs(obj.lane_id) - abs(self.gl.adv.lane_id) ) == 1:
                    self.gl.right_neib_bake_vec_to_compete = obj
                    num += 1
                if num >= 6:
                    break
        lon_warn, lat_warn = False, False
        print("ADV--->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("ADV---adv:",self.gl.adv.name)
        # 前碰撞
        if self.gl.front_vec_to_compete is not None:
            print("ADV---front_vec_to_compete:", self.gl.front_vec_to_compete.name)

        if self.gl.left_neib_front_vec_to_compete is not None:
            print("ADV---left_neib_front_vec_to_compete:", self.gl.left_neib_front_vec_to_compete.name)

        if self.gl.right_neib_front_vec_to_compete is not None:
            print("ADV---right_neib_front_vec_to_compete:", self.gl.right_neib_front_vec_to_compete.name)

        # 后碰撞
        if self.gl.bake_vec_to_compete is not None:
            print("ADV---bake_vec_to_compete:", self.gl.bake_vec_to_compete.name)

        if self.gl.left_neib_bake_vec_to_compete is not None:
            print("ADV---left_neib_bake_vec_to_compete:", self.gl.left_neib_bake_vec_to_compete.name)

        if self.gl.right_neib_bake_vec_to_compete is not None:
            print("ADV---right_neib_bake_vec_to_compete:", self.gl.right_neib_bake_vec_to_compete.name)
            # 计算辆车安全距离
        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)
    def check_lane_change_action(self,new_preceding,new_following):
        # adv vehicle coordination
        if (new_preceding and abs(new_preceding.pos_x) < 6) or (new_following and new_following.name != 'Ego' and  abs(new_following.pos_x) < 6):
            return False
        threshold = 4
        ttc_p = 10
        ttc_f = 10
        if   ( new_preceding  and new_preceding.vx < 2):
            ttc_p =new_preceding.pos_x / new_preceding.vx   if np.fabs(new_preceding.vx ) > 1e-10 else 0.1
        if new_following and new_following.name != 'Ego' and  new_following.vx > -1:
            ttc_f = new_following.pos_x / new_following.vx  if np.fabs(new_following.vx ) > 1e-10 else 0.1
        return abs(ttc_p) > threshold and abs(ttc_f) > threshold
            # 主循环函数
    def ba_func(self,obj,ego_vx,flag = False):
        if flag:
            down_boundary = self.gl.adv_kp_pos[0]
            up_boundary = self.gl.adv_kp_pos[0]
        else:
            down_boundary = obj.initial_state2ego["pos_x"]*0.9
            up_boundary = obj.initial_state2ego["pos_x"]*1.1
        real_vx = ego_vx + obj.vx
        
        if real_vx > 0:
            if obj.pos_x < down_boundary:
                if  obj.pos_x < 0 and obj.pos_x > -10 and  obj.lane_id == self.gl.ego.lane_id:
                    self.gl.scp.dacc(obj.name,0)
                    # print("-----------00------------- speed :",0)
                elif obj.pos_x < -30 and obj.lane_id == self.gl.ego.lane_id:
                    self.gl.scp.dacc(obj.name,self.get_speed(self.gl.ego.vx,self.gl.ego.vy))
                    # print("-----------keep------------ speed :",self.get_speed(self.gl.ego.vx,self.gl.ego.vy))
                else:
                    # print("-----------acc------------ speed :",self.get_speed(self.gl.ego.vx,self.gl.ego.vy)  +2)
                    self.gl.scp.dacc(obj.name,self.get_speed(self.gl.ego.vx,self.gl.ego.vy) + 2)
            elif obj.pos_x > up_boundary:
                self.gl.scp.dacc(obj.name, self.get_speed(self.gl.ego.vx, self.gl.ego.vy) - 2)
            else:
                # print("-----------dacc------------ speed :",self.get_speed(self.gl.ego.vx,self.gl.ego.vy))
                self.gl.scp.dacc(obj.name, self.get_speed(self.gl.ego.vx, self.gl.ego.vy))
        else:
            self.gl.scp.dacc(obj.name, self.get_speed(self.gl.ego.vx, self.gl.ego.vy))

    def ba_exec(self):
        if self.gl.ego.simFrame % 1 == 0:
            ego_vx,ego_vy = trans2angle(self.gl.ego.vx, self.gl.ego.vy,self.gl.ego.pos_h)
            if self.gl.adv:
                if abs(self.gl.adv.pos_x) > 30:
                    print("=============start too slow!!!===============")
                self.ba_func(self.gl.adv,ego_vx,True)
            for obj in self.gl.objects:
                if obj.static:
                    continue
                self.ba_func(obj,ego_vx)
    def initialise(self):
        self.vtd_exec_atomic()
    def update(self):
        # 对抗车控制函数
        # self.vtd_func()
        print("ADV--- Time =================================================== ", time.time() * 1000)
        loop_start_time = time.time()
        # 选定对抗车
        self.vtd_exec_atomic()
        print("vtd_exec_atomic time:", (time.time() - loop_start_time) * 1000)
        # 检查是否到达对抗使能点
        # if not self.check_adv_enable():
        #     # 没到达使能点之前，与主车保持初始状态
        #     self.ba_exec()
        #     return
        # # 非对抗车需一直保持与主车的初始状态，也可以注释掉
        # for obj in self.gl.objects:
        #     if obj.static:
        #         continue
        #     self.ba_func(obj,self.gl.ego.vx)
        # 检查adv_flag为True，则对抗车未追上主车，放弃对抗
        # if self.gl.adv and  not self.gl.adv_flag:
        #     print("======================adv process start too slow~!!!!! stop adv !!!!!=========================")
        #     self.stop(self.gl.adv.id)
        #     self.sendctrlmsg()
        #     return
        # 全局坐标系
        self.trans_world_cood()
        for i in self.gl.objects:
            print("-----------------1---------------------")
            i.show()
            print("-----------------2---------------------")
        if self.gl.adv is not None:

            print("ADV---world cood self.gl.adv:")
            self.gl.adv.show()
            # state, ita_state, ped_state = self.get_dqn_state()
            # self.ctrl_signal = self.sample_actions(state, ita_state, ped_state)
            self.trans_world_to_local()
            print("adv cood    ego info :")
            self.gl.ego.show()
            for i in self.gl.objects:
                print("==================1====================")
                i.show()
                print("==================2====================")
            self.get_adv_around_state()
            # 获取模型输入
            state = self.get_dqn_state_new()
        return py_trees.common.Status.RUNNING
        # other npc
        # if self.gl.adv is None or len(self.gl.objects) > 0:
        # print('npc size : ', len(self.gl.objects))
        # if  len(self.gl.objects) > 0:
        #     self.vtd_func()
        # if 0:
        if self.gl.adv is not None:
            # print("ADV>>>>>>>>>>>>>>>>>:",self.gl.adv.show())
            # 打开对抗车所有大灯，标记选定对抗车
            if self.gl.compete_time % 10 == 1:
            # if self.gl.compete_time  == 1:
                self.gl.scp.vec_light(self.gl.adv.name, left=True, right=True)

            # 如果横向动作预留时间即将用完，当前帧切换为自动驾驶模式
            if self.keep_time > 0 and  self.keep_time_index + 1 == self.keep_time:
                self.keep_time_index += 1
                # print("ADV--->>>>>>>>>>>>>11111>>>>>>>>>>>",self.keep_time_index, self.keep_time)
                self.autopilot(self.gl.adv.id)
            else:
                # print("================================================")
                # self.gl.ego.show()
                # self.gl.adv.show()
                # print("================================================")
                # 从全局坐标系转换为对抗车车辆坐标系
                self.trans_world_to_local()
                # 获取对抗车周围目标物状态
                self.get_adv_around_state()
                # 获取模型输入
                state = self.get_dqn_state_new()
                # 获取模型输出指令
                self.ctrl_signal = self.sample_actions_new(state)
                # self.ctrl_signal = {"lon":"IDLE","lat":"RIGHT_2"}
                # LEFT_2
                print("ADV--- model ctrl:", self.ctrl_signal)
                # 碰撞检测，如果有碰撞危险返回True, 此处会将所有目标转换为对抗车辆坐标系
                lon_warn , lat_warn = self.collision_warning()
                if lon_warn:
                    # 刹车
                    self.stop(self.gl.adv.id)
                else:
                    # 控制指令
                    self.contrl(lat_warn)
            # ego colision adv
            self.collision_check(self.gl.adv, self.gl.ego)
            self.gl.last_compete_name = self.gl.adv.name
            self.gl.last_compete_name_id = self.gl.adv.id

        # 释放对抗车控制权
        if self.gl.adv is None   and  self.gl.last_compete_name != '' or (self.gl.adv is not None and self.gl.last_compete_name != '' and   self.gl.adv.name != self.gl.last_compete_name) :
            self.gl.scp.vec_light(self.gl.last_compete_name, left=False, right=False)
            # 释放掉之前对抗车控制权
            self.autopilot(self.gl.last_compete_name_id)
            self.gl.compete_time = 0
        self.sendctrlmsg()
        # if save_reference_time:
        #     data = np.array([[  time.time() - loop_start_time  ]])
        #     df1 = pd.DataFrame(data=data,columns=env_key)
        #     df = df.append(df1, ignore_index=True)
            # df = pd.concat(df,df1,ignore_index=True) df.append(df1, ignore_index=True)


        print("ADV---loop_total_time:", time.time() - loop_start_time)
        # df.to_csv('./reference_time' + '.csv', encoding='utf_8_sig')
