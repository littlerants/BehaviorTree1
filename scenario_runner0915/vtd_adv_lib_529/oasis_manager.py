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
from gym_sumo.algo.global_route_planner_vtd_xodr import GlobalRoutePlanner
from gym_sumo.road.road import Road, RoadNetwork
from gym_sumo.road.lane import LineType, PolyLaneFixedWidth

from vtd_adv_lib.global1 import GLOBAL
from vtd_adv_lib.model_namager import ModelManager
from vtd_adv_lib.scenario import SCENARIO, PREPARE
from vtd_adv_lib.object import OBJECT
from vtd_adv_lib.utils import *
from vtd_adv_lib.head import *
from common.misc import get_speed, calculate_distance
from srunner.scenariomanager.carla_data_provider import (
    CarlaDataProvider,
)
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
class VTD_Manager:
    # 速度阶梯查询表，用来计算对抗车预期加速度，假设车辆行驶速度范围0-30
    DEFAULT_TARGET_SPEEDS = np.linspace(0, 30, 15)
    
    def __init__(self, open_vtd = False,vtdPort = 48190,sensorPort = 48195, plugins_path = 'data/lib/sendTrigger0109.so',test = False,args = None):

        self.open_vtd = open_vtd
        self.scenario = SCENARIO()
        args.env_config['map_file_path'] = self.scenario.xodr_path
        self.args = args

        if test:
            self.model_manager =  ModelManager(self.args ,self.args.output_path)
        if self.open_vtd:
            # prepare
            # 需加载xodr，获取道路信息，并结构化道路
            self.map = GlobalRoutePlanner(self.scenario.xodr_path)
            self.road = self.make_road(self.map)
            # 相遇点
            self.meeting_points = None
            # 所有juction道路 set
            self.junction_edge_list = set()
            self.junction_road_info = dict()
            self.juction_set()
            # 车辆是否在juction中
            self.in_juction = False
            # self.close_light = True
            # 预处理对象，在场景运行前，主要用来获取xodr或者 xml中一些信息，比如相遇点位置等
            self.prepare = PREPARE(self.scenario.scenario_path)

            # ACTIONS =  {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"}
            # 强化学习模型动作空间中各个动作执行时间
            self.ACTIONS_DUR =  { 'LANE_LEFT': 80, 'IDLE': 0 , 'LANE_RIGHT': 80, 'FASTER': 10, 'SLOWER': 10,
                                "LEFT_1": 50, "LEFT_2": 50, "RIGHT_1": 50, "RIGHT_2": 50}
            self.action_marking = ''
            # 各个动作预留时间         
            self.keep_time = -1

            # 各个动作已用动作时间
            self.keep_time_index = -1
            # self.disappear_num = 0
            # 控制指令
            self.ctrl_signal = None
            # 安全距离
            self.safe_dis  = 4
            # 真实距离
            self.dis = 0
            ##############################################################################################
            ##############################################################################################
            ##############################################################################################
            # 加载c++库，用来给对抗车发送纵向/横向指令，源文件请转至 your_project_path/ca/RDBClientSample/sendtrigger.cpp
            self.ll = ctypes.cdll.LoadLibrary
            # print("os.getcwd() + plugins_path:",os.getcwd() + plugins_path)
            os_path = os.path.join( os.getenv('ADVPATH'), plugins_path)
            print('os_path:',os_path)
            # self.lib = self.ll(os.getcwd() + plugins_path)
            self.lib = self.ll(os_path)
            self.lib.addPkg.argtypes = [ctypes.c_double, ctypes.c_uint32 ,ctypes.c_double,ctypes.c_double,ctypes.c_uint32,ctypes.c_uint32]
            self.lib.addPkg.restype =  ctypes.c_int
            self.lib.connect1.argtypes = [ctypes.c_int]
            self.lib.connect1.restype =  ctypes.c_int
            self.lib.get_msg_num.argtypes = []
            self.lib.get_msg_num.restype =  ctypes.c_int
            #  sendSocket, simTime,  simFrame,  send_trigger = 1
            self.lib.sendTrigger.argtypes = [ctypes.c_int,ctypes.c_double, ctypes.c_uint32, ctypes.c_int]
            self.lib.sendTrigger.restype =  ctypes.c_int
            self.sClient =  self.lib.connect1(vtdPort)

            self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
            self.tcp_server.connect(("127.0.0.1",sensorPort))
            
            ##############################################################################################
            ##############################################################################################
            ##############################################################################################

            # 初始化全局对象
            self.gl = GLOBAL()
            # self.gl.get_confrontation()

            # RDB message 接受字节数
            self.ret = 0
            # 剩余未处理字节数
            self.bytesInBuffer = 0
            # 当前处理字节位置（指针）
            self.pData = 0
            # message 报文头解析对象
            self.handle = HANDLE()

            
            # 加速度平滑处理-----巴特沃斯低通滤波器
            self.listdp = deque(range(10),maxlen=10)
            self.N  = 3   # Filter order
            self.Wn = 0.1 # Cutoff frequencyself.B
            self.B = None
            self.A = None
            self.B, self.A = signal.butter(self.N, self.Wn, output='ba')
            # 检测加速度正负反转状态
            self.C_acc = []
            self.C_time = -1
            self.C = 0
            # carla ego
            self.ego_vehicle = None
            self.world = None
            self._map = None
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
                    print(edge_id)
                lane_id.add(edge_id)
                net.add_lane(edge_id[0], edge_id[1],
                                PolyLaneFixedWidth(center_lines,
                                                start_wpt.width,
                                                lane_index=edge_id,
                                                line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)))

        return  Road(network=net, np_random=1, record_history=False)

    def get_sqrt(self,x,y):
        return math.sqrt(x**2 + y**2)

    def chek_in_juction(self,roadId,section, lane_id):
        in_juction = False
        # print("11111:",roadId,section, lane_id)
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
        return  other_vehicles

        # 初始化或更新目标对象
    def create_objs(self):
        # self.gl.objects_set 所要维护的目标列表
        # 若为0，则直接清空 self.gl.objects（用于存储所有目标对象）
        objects,traffic_lights  = self.update_info()
        ego_wp = self._map.get_waypoint( self.ego_vehicle.get_location() )
        # 车辆朝向 左手坐标系
        pos_h  = self.ego_vehicle.get_transform().rotation.yaw
        # 道路朝向
        hdg = ego_wp.transform.rotation.yaw

        if len(self.gl.objects_set) == 0:
            self.gl.objects.clear()
        # 如果主车不为空，更新主车各个参数
        if self.gl.ego is not None:

            self.gl.ego.update( pos_x= self.ego_vehicle.get_location().x , pos_y= self.ego_vehicle.get_location().y,pos_h = self.ego_vehicle.get_transform().rotation.yaw ,\
            vx = self.ego_vehicle.get_velocity().x, vy = self.ego_vehicle.get_velocity().y, acc_x= self.ego_vehicle.get_acceleration().x, hdg= ego_wp.transform.rotation.yaw,\
            l= self.ego_vehicle.bounding_box.extent.x, w= self.ego_vehicle.bounding_box.extent.y ,acc_y= self.ego_vehicle.get_acceleration().y, roadId=ego_wp.road_id,
            obj_type=self.ego_vehicle.type_id,lane_offset= 0 ,lane_id=ego_wp.lane_id, \
            leftLaneId = ego_wp.get_left_lane().lane_id if  ego_wp.get_left_lane() else None , rightLaneId = ego_wp.get_right_lane().lane_id if ego_wp.get_right_lane() else None,\
            light_state = traffic_lights
            )
        # 否则初始化主车对象
        else:
            self.gl.ego = OBJECT(name=self.ego_vehicle.id,\
            id=self.ego_vehicle.id, pos_x=self.ego_vehicle.get_location().x ,off_x=0, pos_y=self.ego_vehicle.get_location().y, pos_h = self.ego_vehicle.get_transform().rotation.yaw,\
            hdg= ego_wp.transform.rotation.yaw ,  vx = self.ego_vehicle.get_velocity().x, vy=self.ego_vehicle.get_velocity().y,roadId=ego_wp.road_id ,\
            acc_x=self.ego_vehicle.get_acceleration().x, l=self.ego_vehicle.bounding_box.extent.x, w=self.ego_vehicle.bounding_box.extent.y ,\
            acc_y=self.ego_vehicle.get_acceleration().y ,obj_type = self.ego_vehicle.type_id,lane_offset=0,\
            lane_id=ego_wp.lane_id, leftLaneId=ego_wp.get_left_lane().lane_id  if  ego_wp.get_left_lane() else None , rightLaneId =  ego_wp.get_right_lane().lane_id if ego_wp.get_right_lane() else None,
            )
        # 遍历self.gl.fram_data['Objects'] 中探测到的目标，并初始化或更新这些目标
        for i in objects:
            
            # Npc车辆离路保护分支
            # if 'lane_id' in i:
            #     c = i['lane_id']
            # else:
            #     print("object ", i['name'] ," is not on road!!!!!!!!!!!!!!1")
            #     tmp_lane_id = 0
            #     continue
            # 如果在维护列表中，更新状态，否则初始化对象
            i_wp = self._map.get_waypoint(i.get_location())
            if i.id in self.gl.objects_set:
                for j in self.gl.objects:
                    if i.id== j.id:
                        j.update( pos_x= i.get_location().x, pos_y=i.get_location().y,pos_h = i.get_transform().rotation.yaw,hdg=i_wp.transform.rotation.yaw ,\
                        vx=i.get_velocity().x, vy=i.get_velocity().y,
                        acc_x=i.get_acceleration().x, l=i.bounding_box.extent.x, w=i.bounding_box.extent.y,acc_y=i.get_acceleration().y,obj_type=i.type_id,\
                        lane_offset= 0,lane_id = i_wp.lane_id ,roadId=i_wp.road_id,\
                        leftLaneId = i_wp.get_left_lane().lane_id  if  i_wp.get_left_lane() else None , rightLaneId = i_wp.get_right_lane().lane_id if ego_wp.get_right_lane() else None
                            )
            else:
                obj = OBJECT(id=i.id, pos_x= i.get_location().x, pos_y=i.get_location().y,off_x= 0 ,pos_h = i.get_transform().rotation.yaw,hdg=i_wp.transform.rotation.yaw ,\
                             vx=i.get_velocity().x, vy=i.get_velocity().y,
                acc_x=i.get_acceleration().x, l=i.bounding_box.extent.x, w=i.bounding_box.extent.y,acc_y=i.get_acceleration().y,obj_type=i.type_id,\
                lane_offset= 0,lane_id = i_wp.lane_id ,roadId=i_wp.road_id, \
                leftLaneId=i_wp.get_left_lane().lane_id if i_wp.get_left_lane() else None,
                rightLaneId=i_wp.get_right_lane().lane_id if ego_wp.get_right_lane() else None
                             )
                self.gl.objects_set.append(obj.id)
                self.gl.objects+= [obj]      
        # 计算各个车辆与主车的方位 ，会根据这些方位制定对应的策略
        for   i in self.gl.objects:
            # print(i.name)
            # i.pos_trajx.append(i.pos_x)
            # i.pos_trajy.append(i.pos_y)
            # 如果在主车前方3米，则标记为0
            if i.pos_x > 3:
                i.direction_to_ego = 0
            # 如果在主车后方-1米，则标记为1
            elif i.pos_x < -1:
                i.direction_to_ego = 1
            # 左侧，为2
            elif i.pos_y > 1:
                i.direction_to_ego = 2
            # 右侧，为3
            elif i.pos_y < -1:
                i.direction_to_ego = 3
    def get_sqrt(self,x,y):
        return math.sqrt(x**2 + y**2)
    def parseRDBMessage(self,pData, data,handle):

        #  创建对象
        self.create_objs()

        # 根据与主车的距离远近，对所有探测到的目标物进行排序
        self.gl.objects = sorted(self.gl.objects, key=lambda x: math.sqrt(x.pos_x**2 + x.pos_y**2) )
        flag = {'index':0,"flg":False}
        # 将要从 self.objects中弹出的对象列表
        pop_index = []
        index = 0
        # 与主车最近的一辆车的index，若对抗车存在且不是最近的目标物，则此目标作为备选对抗车辆
        front_close_vec_index = -1
        
        show_log = 0
        if show_log:
           print("len(self.gl.objects):",len(self.gl.objects))
        if len(self.gl.objects) >0:
            # print("len objects:",len(self.gl.objects))g
            for i in self.gl.objects:
                # ego front vecs
                # print('distance:',self.get_sqrt(i.pos_x, i.pos_y))
                # 主车50米以内所有车辆
                if self.get_sqrt(i.pos_x, i.pos_y) <30:
                    # 主车后30米以内所有车辆
                    if  i.pos_x > -20:
                        # print("i.pos_x > 3")
                        # i.show()
                        # print("self.gl.compete_time:",self.gl.compete_time)
                        # 选定一辆对抗车后，对抗时间持续 500 帧
                        if self.gl.compete_time < self.gl.compete_time_range :
                            
                            if front_close_vec_index == -1:
                                # 获取距离主车最近的目标index
                                front_close_vec_index = index
                                # front_close_vec_index = -1
                            # 如果对抗时长还未用完，且当前目标物与上一帧目标物为同一目标物，则被优先选择
                            if i.name ==  self.gl.last_compete_name:
                            # if i.name ==  'adv':
                                pop_index.append(index)
                                flag['flg'] = True
                                flag['index'] = index
                        # 对抗时间用完，则清空所有对抗信息，重新选择对抗目标
                        else:
                            # auto pilot
                            self.gl.compete_time = 0
                            self.gl.last_compete_name = ''
                            self.gl.last_compete_name_id = -1
                          #  print("comete time is over!!!")
                            return None
                # else:
                #     pop_index.append(index)

                index += 1

        else:
            # print()
            self.gl.compete_time = 0
            print("ego near no compete vecs !!!")
            return None
        
        if show_log:
           print("flag::",flag)
           print("front_close_vec_index:",front_close_vec_index)
        # 更新对抗车辆
        if flag['flg'] :
            if self.gl.adv is not  None and  self.gl.adv.name == self.gl.objects[flag['index']].name:
                self.gl.adv.update1( self.gl.objects[flag['index']])
            else:
                self.gl.adv =self.gl.objects[flag['index']]
        elif  front_close_vec_index != -1:
            self.gl.compete_time = 0
            pop_index.append(front_close_vec_index)
            self.gl.adv = self.gl.objects[front_close_vec_index]
            # self.gl.objects.pop(front_close_vec_index)
            # self.gl.objects_set.remove(self.gl.objects[front_close_vec_index].id)
        else:
          #  print("ego front no compete vecs !!!")
            self.gl.adv = None
            self.gl.compete_time = 0
            return None
        # print("self.gl.adv.name:",self.gl.adv.name)
        # add lane object
        lane = None
        # 更新lane对象
        section = 0
        if self.gl.adv is not None:
            self.gl.compete_time += 1
            # lane_index = self.map.road_id_to_edge[ self.gl.adv.roadId ][ section ][ self.gl.adv.lane_id ]
            # if self.gl.adv.lane is  None  or  (lane_index !=  self.gl.adv.lane.lane_index) :
            #     lane_index = lane_index + (0,)
            #     try:
            #         lane = self.road.network.get_lane(lane_index)
            #     except:
            #         pass
            #     self.gl.adv.update(lane=lane)
        
        # 从目标物维护列表中删除对抗车，以及较远的目标
        pop_index  =sorted(pop_index,reverse=True)
        # print("pop index:",pop_index)
        # print("self.gl.objects_set:",self.gl.objects_set)
        # for i in self.gl.objects:
        #     i.show()
        # print("88888888888888888888888888")
        for p in pop_index:
            # print("p:",p)
            # print("self.gl.objects[p].id:",self.gl.objects[p].name,self.gl.objects[p].id)
            self.gl.objects_set.remove(self.gl.objects[p].id)
            self.gl.objects.pop(p)
        if show_log:
            # print("len objs:",len(self.gl.objects))
            print("self.gl.adv:")
            self.gl.adv.show()
            print("self.gl.ego:")
            self.gl.ego.show()
        # 当前函数就是处理仿真世界当前帧数据并选取对抗车辆
            
    # 获取车辆距离
    def get_distance(self,ego):
        return math.sqrt( math.pow(ego.pos_x,2) + math.pow( ego.pos_y,2) )
    # 获取车辆速度
    def get_speed(self,vx,vy):
        return math.sqrt( math.pow(vx,2) + math.pow( vy,2) )

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
        return np.int64(np.clip(
            np.round(x * (self.DEFAULT_TARGET_SPEEDS.size - 1)), 0, self.DEFAULT_TARGET_SPEEDS.size - 1))
    # speed m/s
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
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0.1, 0, self.gl.adv.id  , 1)
    # 右变道
    def lane_right_act(self):
        self.gl.scp.turn_right(self.gl.adv.name)

    # 加速指令
    def faster_act(self,var = 1):
        # 此时坐标系已转换为对抗车坐标系下
        # 对抗车绝对速度
        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)
        # print("adv_speed:",adv_speed)
        # print(self.gl.adv.vx,self.gl.adv.vy)
        # 获取预期速度索引
        # 获取目标速度 target_speed
        speed_index = self.get_speed_index(adv_speed)
        if adv_speed < self.DEFAULT_TARGET_SPEEDS[ 1 ] and var == -1:
            target_speed = self.DEFAULT_TARGET_SPEEDS[ 0 ] + 1
        elif adv_speed > self.DEFAULT_TARGET_SPEEDS[ -1 ] and var == 1:
            target_speed = adv_speed + 3
        else:
            # 防止索引溢出
            if speed_index >= len(self.DEFAULT_TARGET_SPEEDS):
                var = 0
            target_speed = self.DEFAULT_TARGET_SPEEDS[ speed_index + var ]
        # 预期加速度
        acceleration = target_speed  - adv_speed

        # 获取 对抗车与主车的距离
        self.dis = self.get_distance(self.gl.ego)
        # 速度较大，加速度会添加偏置var
        if speed_index > 3:
            var = 1 - speed_index / 15
            acceleration = (acceleration + var) if acceleration > 0 else acceleration - var
        # 平滑处理
        self.listdp.append(acceleration)
        if len(self.listdp)  == 13:
            alpha = 0.2
            temp = np.array(self.listdp)
            temp = temp.astype(np.float)
            tempf = signal.filtfilt(self.B, self.A, temp)
                        
            ema = [tempf[0]]  # 初始EMA值等于第一个数据点
            for i in range(1, len(tempf)):
                ema.append(alpha * tempf[i] + (1 - alpha) * ema[-1])
          #  print(ema[-1])
            acceleration = tempf[-1]

        # 获取加速度最大最小值
        max_acc = self.get_speed(self.gl.adv.vx + self.gl.ego.vx,self.gl.adv.vy + self.gl.ego.vy) / 10 
        min_acc = self.get_max_acc(adv_speed)
        if max_acc < 4:
            max_acc = 4
        # 在变道等动作前半段时，抑制减速度
        if self.keep_time_index > 0 and self.keep_time_index < (self.keep_time>>1):
            max_acc = 4
            min_acc = -3
        
        # print("max_acc:",max_acc)
        if acceleration > max_acc:
            acceleration = max_acc
        if acceleration < min_acc:
                acceleration = min_acc

        # 加速度正负反转状态
        acceleration = self.opt_conver_acc(acceleration)
        # G
        # if self.gl.ego.pos_x < 0 and  abs(self.gl.adv.vx + self.gl.ego.vx) < 10 and dis < 13:
        #     acceleration = 0.5
        # 计算辆车安全距离
        if self.gl.open_lat:
            self.safe_dis = (self.gl.adv.vx + self.gl.ego.vx)*0.1 + 4 + math.pow(self.gl.ego.vx,2) / (2*(-self.get_max_acc(adv_speed) - 0.03* ((self.gl.adv.vx + self.gl.ego.vx)/10)   ) )
            # if self.dis < self.safe_dis and self.gl.adv.lane_id != self.gl.ego.lane_id:
            # 安全距离 + 4 是修正车身长度， 如果辆车距离小于安全距离，并且辆车纵向速度差大于2
            if self.dis < ((self.safe_dis + 5) if self.safe_dis < 5 else 11) :
                acceleration = 1 + abs(self.gl.ego.vx) if abs(self.gl.ego.vx) < 1 else 1 - self.dis / (self.safe_dis + 5)
        print("safe_dis:", (self.safe_dis ))
        print("dis:",self.dis)
        # # print("self.gl.ego.pos_x:",self.gl.ego.pos_x)
        # if self.gl.ego.pos_x < 0 and abs(self.gl.adv.vx + self.gl.ego.vx) < 10 and   dis < (safe_dis if safe_dis > 10 else 10):
        #     acceleration = 1
        # if self.in_juction:
        #     if adv_speed >= self.get_speed(self.gl.adv.vx + self.gl.ego.vx,self.gl.adv.vy + self.gl.ego.vy)* 2:
        #         acceleration = 0 
        # 添加加速pkg
        self.gl.target_acc = acceleration
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], acceleration, 0, self.gl.adv.id  , 1)

    def opt_conver_acc(self,acceleration):
        # print("self.C_time:",self.C_time)
        # print("self.C_time:",self.C)
        # print("acceleration:",acceleration)
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
        
        # print("self.C_acc:",self.C_acc)
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
            print("left 车道偏离")
            # self.right_1_act()
            self.LCA()
            
        if abs(self.gl.adv.rightLaneId ) > 30 and self.action_marking in ['RIGHT_2','LANE_RIGHT']:
            print("right 车道偏离")
            self.action_marking ='LEFT_1'
            self.LCA()
            # # self.left_1_act()

    def exec_lat_act(self):
        # self.ctrl_signal['lat'] = "RIGHT_2"
        if self.get_speed( self.gl.adv.vx, self.gl.adv.vy) <  3:
            return
        gain = 0
        if self.dis < self.safe_dis :
            gain = 0.5
            # if self.gl.ego.lane_id == self.gl.adv.lane_id:
            #     return
            
        left_lane  = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
        right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
        #  主车在对抗车右侧，禁止左方向移动
        if self.gl.ego.pos_y < 0:
            left_lane = 0
        if self.gl.ego.pos_y > 0:
            right_lane = 0
        # lane_off_set = abs(self.gl.ego.pos_y) - self.gl.ego.w/2 - self.gl.adv.w/2
        # lane_off_set = lane_off_set if lane_off_set > 1.8  else 2
        # 如果左侧或者右侧存在lane，则允许较大偏移
        if left_lane or right_lane:
            lane_off_set = 1.7
        else:
            lane_off_set = 1.4
        # lane_off_set = 0.5
        # lane_off_set -= gain
        if self.ctrl_signal['lat']  == "LANE_LEFT" and left_lane :
            self.lane_left_act()
        elif self.ctrl_signal['lat'] == "LEFT_1" :
            self.left_1_act()
        elif self.ctrl_signal['lat'] == "LEFT_2" :
            self.left_2_act(lane_off_set)

        elif self.ctrl_signal['lat'] == "LANE_RIGHT" and right_lane :
            self.lane_right_act()
        elif self.ctrl_signal['lat'] == "RIGHT_1":
            self.right_1_act()
        elif self.ctrl_signal['lat'] == "RIGHT_2" :
            self.right_2_act(lane_off_set)
        else:
            self.gl.scp.auto(self.gl.adv.name)
            self.keep_time = -1
            self.keep_time_index = -1

    def exec_lat_act(self,act):
        # self.ctrl_signal['lat'] = "RIGHT_2"
        if self.get_speed(self.gl.adv.vx, self.gl.adv.vy) < 3:
            return
        gain = 0
        if self.dis < self.safe_dis:
            gain = 0.5
            # if self.gl.ego.lane_id == self.gl.adv.lane_id:
            #     return

        left_lane = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
        right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
        #  主车在对抗车右侧，禁止左方向移动
        if self.gl.ego.pos_y < 0:
            left_lane = 0
        if self.gl.ego.pos_y > 0:
            right_lane = 0
        # lane_off_set = abs(self.gl.ego.pos_y) - self.gl.ego.w/2 - self.gl.adv.w/2
        # lane_off_set = lane_off_set if lane_off_set > 1.8  else 2
        # 如果左侧或者右侧存在lane，则允许较大偏移
        if left_lane or right_lane:
            lane_off_set = 1.7
        else:
            lane_off_set = 1.4
        # lane_off_set = 0.5
        # lane_off_set -= gain
        if act == "LANE_LEFT" and left_lane:
            self.lane_left_act()
        elif act== "LEFT_1":
            self.left_1_act()
        elif act == "LEFT_2":
            self.left_2_act(lane_off_set)

        elif act == "LANE_RIGHT" and right_lane:
            self.lane_right_act()
        elif act == "RIGHT_1":
            self.right_1_act()
        elif act == "RIGHT_2":
            self.right_2_act(lane_off_set)
        else:
            self.gl.scp.auto(self.gl.adv.name)
            self.keep_time = -1
            self.keep_time_index = -1
    def slow_following_check(self):
        # 慢跟车处理逻辑
        adv_speed = self.get_speed(self.gl.adv)
        result = None
        # 如果对抗车速度小于3m/s，则抑制掉横向对抗动作，此处不影响慢车换道逻辑，result会在满足慢车换道逻辑后重新刷新
        if self.dis < ((self.safe_dis + 5) if self.safe_dis < 5 else 11) :
            result =  'IDLE'

        if  adv_speed < 1.5:
            if self.gl.adv_pre_speed < 1.5:
                # 如果主车当前帧和上一帧都是慢速，slow_dur加一
                self.gl.slow_dur += 1  # 在GLOBAL里添加属性
                # 如果慢速跟车持续300 帧以上，则执行换道指令
                if self.gl.slow_dur > 300:
                    # 对抗车左侧车道可通行，优先左边到
                    if abs(self.gl.adv.leftLaneId) < 30 :
                        self.lane_left_act()
                        result = "LANE_LEFT"
                    elif abs(self.gl.adv.rightLaneId) < 30:
                        result = "LANE_RIGHT"
                    else:
                        pass
                    # 如果计时大于300，也需要清零
                    self.gl.slow_dur = 0
        # 当速度大于1.5时，一定要清零计时
        elif adv_speed >= 1.5:
            self.gl.slow_dur = 0
        self.gl.adv_pre_speed = adv_speed
        return result


    def contrl_adv(self):
        # ACTIONS =  {0: 'LANE_LEFT', 1: 'IDLgE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"}
        #     self.ACTIONS_DUR =  { 'LANE_LEFT': 40, 'IDLE': 0 , 'LANE_RIGHT': 40, 'FASTER': 10, 'SLOWER': 10,
        #                         "LEFT_1": 20, "LEFT_2": 20, "RIGHT_1": 20, "RIGHT_2": 20}
        # print("self.ctrl_signal:",self.ctrl_signal)
        #对抗目标若是行人，走此分支 
        if self.gl.adv.obj_type == 5:
            adv_speed  =  self.gl.adv.get_sqrt(self.gl.adv.vx, self.gl.adv.vy)
            if self.ctrl_signal['lon'] == 'FASTER':
                if adv_speed > 2:
                    adv_speed = 2
                else:
                    adv_speed += 1
                self.gl.scp.dacc(actor=self.gl.adv.name, target_speed = adv_speed , type=5  )
            # self.gl.scp.dacc(self,actor, target_speed = 20,type = None):
            elif self.ctrl_signal['lon'] == 'SLOWER':
                if adv_speed < 0.3:
                    adv_speed = 0.3
                else:
                    adv_speed  -= 0.5
                self.gl.scp.dacc(actor=self.gl.adv.name, target_speed = adv_speed , type=5  )
            elif self.ctrl_signal['lon'] == 'IDLE':
                pass
            return

        # 慢车处理
        # 如果对抗车速度小于3m/s,此时VTD中横向动作不自然，抑制掉横向动作

        slow_check = self.slow_following_check()
        if slow_check is not None:
            print("slow ")
            self.ctrl_signal['lat'] = slow_check
            if slow_check != 'IDLE':
                self.ctrl_signal['lon'] = 'FASTER'

        # 纵向控制指令
        if self.ctrl_signal['lon'] == 'FASTER':
            self.faster_act()
        elif self.ctrl_signal['lon'] == 'SLOWER':
            self.slower_act()
        elif self.ctrl_signal['lon'] == 'IDLE':
            self.idle_act()
        # 横向控制指令


        if self.keep_time <= 0:
            self.keep_time = self.ACTIONS_DUR[self.ctrl_signal['lat']]
            self.action_marking = self.ctrl_signal['lat']
            if self.gl.open_lat:
                self.exec_lat_act()
        # if self.keep_time_index >= 0 and  self.keep_time_index < self.keep_time:
        #     self.LDWS()
        #     self.keep_time_index += 1
        #     return -1
        elif self.keep_time_index >= self.keep_time:
            # 如果是变道指令，并且预留时间已经用完，但是变道还未完成，则再增加10帧预留时间，预留时间上限为300帧,300帧后强制清零，防止出现不合理现象
            if self.action_marking in ['LANE_RIGHT','LANE_LEFT'] and self.gl.adv.lane_id == self.gl.pre_adv_lane_id and self.keep_time < 300:
                self.keep_time += 10
            else:
                self.keep_time = -1
                self.keep_time_index = -1
                self.action_marking = ''
                self.gl.scp.overLaneoffset(self.gl.adv.name)
            return -1
        self.keep_time_index += 1
        if self.ctrl_signal['lat'] == 'IDLE':
            self.action_marking = 'IDLE'
            return
 

    # 处理一帧数据
    def vtd_exec_atomic(self):
        self.parseRDBMessage()

    def get_sqrt(self,x,y):
        return math.sqrt(math.pow(x,2) + math.pow(y,2))

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(1, self.args.state_dim), low=-np.inf, high=np.inf,
                          dtype=np.float32)
    
    # def check_inter(self):
    #     start  = time.time()
    #     if random_edge not in interaction_lane_id.keys():
    #         interaction_lane_id[random_edge] = []
    #         interaction_point_id[random_edge] = dict()
    #         ego_lane = road.network.get_lane(random_edge + (0,))
    #         ego_center = np.array(ego_lane.map_vector)[:, :2].tolist()
    #         ego_lane_width = ego_lane.map_vector[0][2] - 0.1
    #         for bv_edge_id in list(junction_edge_list):
    #             if random_edge == bv_edge_id:
    #                 continue
    #             bv_lane = road.network.get_lane(bv_edge_id + (0,))
    #             bv_center = np.array(bv_lane.map_vector)[:, :2].tolist()
    #             bv_lane_width = bv_lane.map_vector[0][2] - 0.1
    #             if np.linalg.norm(np.array(ego_center[0]) - np.array(bv_center[0])) <= 1:
    #                 continue
    #             intersection, intersection_center = check_lane_intersection(ego_center, ego_lane_width, bv_center, bv_lane_width)
    #             if intersection:
    #                 interaction_lane_id[random_edge].append(bv_edge_id)
    #                 interaction_point_id[random_edge][bv_edge_id] = intersection_center[len(intersection_center) // 2]
    #     end_time = time.time()
    
    # 获取强化学习模型输入状态
    def get_dqn_state(self,left_lane = 1,right_lane = 1):
        # 纵向模型输入
        traj_obs = []
        # 交互模型输入，主要处理十字路口，环岛juction场景
        ita_obs = []
        #行人模型输入
        ped_obs = []
        
        ego_speed = self.gl.ego.get_sqrt(self.gl.ego.vx, self.gl.ego.vy) 
        adv_speed = self.gl.adv.get_sqrt(self.gl.adv.vx, self.gl.adv.vy)
        if self.gl.adv is not None:
            left_lane  = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
            right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
            adv_hdg = self.gl.adv.pos_h
            adv_lane_hdg = self.gl.adv.pos_h  + self.gl.adv.hdg 
            if adv_hdg > 3.14:
                adv_hdg -= 6.28
                adv_lane_hdg -=  6.28
            #模型输入 [对抗车纵向速度，对抗车横向速度，对抗车航向角，对抗车车道偏移，对抗车所在道路航向角，对抗车是否可以左转 ， 对抗车是否可以右转]
            traj_obs = [self.model_manager.normalize_obs( self.gl.adv.vx,'vx'),\
                 self.model_manager.normalize_obs(self.gl.adv.vy,'vy'), \
                 adv_hdg, self.gl.adv.lane_offset, adv_lane_hdg, left_lane, right_lane]
            ita_obs +=  traj_obs
            ped_obs += traj_obs
        
        
        # 如果车辆在juciton中
        if self.in_juction and self.gl.ego.lane is not None and self.gl.adv.lane is not None:
            # 获取主车车道中心点集
            ego_center = np.array(self.gl.ego.lane.map_vector)[:, :2].tolist()
            # 主车车道宽度
            ego_lane_width = self.gl.ego.lane.map_vector[0][2] - 0.1
            # 对抗车车道中心点集
            bv_center = np.array(self.gl.adv.lane.map_vector)[:, :2].tolist()
            # 对抗车车道宽度
            bv_lane_width = self.gl.adv.lane.map_vector[0][2] - 0.1
            # if np.linalg.norm(np.array(ego_center[0]) - np.array(bv_center[0])) <= 1:
            #     return 

            intersection, intersection_center = check_lane_intersection(ego_center, ego_lane_width, bv_center, bv_lane_width)
            # if intersection and self.gl.adv.roadId  != self.gl.ego.roadId   and self.gl.adv.lane.lane_index != self.gl.ego.lane.lane_index:
            if intersection and self.gl.adv.roadId  != self.gl.ego.roadId:
                self.in_juction = True
                # 相遇点
                self.meeting_points = np.array(intersection_center[len(intersection_center) // 2])
                ego_lon, _ = self.gl.ego.lane.local_coordinates(self.meeting_points)
                ego_current_lon,_ = self.gl.ego.lane.local_coordinates(np.array([self.gl.ego.pos_x, self.gl.ego.pos_y]))
                bv_lon, _ = self.gl.adv.lane.local_coordinates(self.meeting_points)
                bv_current_lon , _= self.gl.adv.lane.local_coordinates(np.array([self.gl.adv.pos_x, self.gl.adv.pos_y]))
                # 主车到达相遇点时间
                ego_t = (ego_lon - ego_current_lon) / ego_speed if ego_speed > 0.1 else 0.1
                # 对抗车到达相遇点时间
                bv_t = (bv_lon - bv_current_lon) / adv_speed if adv_speed > 0.1 else 0.1
                # 交互模型添加 时间差维度
                ita_obs.append( bv_t - ego_t )
                # 交互模型添加 距离差维度
                ita_obs.append( bv_lon - bv_current_lon )
            else:
                self.in_juction = False
        # 行人模型
        if self.gl.adv.obj_type == 5  :
            # ex = np.array(self.gl.ego.pos_trajx)
            # ey = np.array(self.gl.ego.pos_trajy)
            # 如果轨迹大于25个点，进入分支，选取最小二乘进行拟合
            if  len(self.gl.adv.pos_trajx) >= 25:
                ay = np.array(self.gl.adv.pos_trajx)
                ax = np.array(self.gl.adv.pos_trajy)
                # ego_p =  np.polyfit(ex,ey,3)
                adv_p =  np.polyfit(ax,ay,3)
                print('************************************')
                # print(np.poly1d(adv_p))
                print(adv_p[-1])
                # self.gl.adv.pos_trajx 目标车的轨迹
                if  self.gl.adv.pos_trajy[-1] > -10 and self.gl.adv.pos_trajx[-1] > 0 and  adv_p[-1] <= 15 and adv_p[-1] > 0:
                    ego_t = adv_p[-1]  / ego_speed if ego_speed > 0.1 else 0.1
                    bv_t = self.gl.adv.pos_trajy[-1] / adv_speed if adv_speed > 0.1 else 0.1
                    ped_obs.append( bv_t - ego_t )
                    ped_obs.append( abs(self.gl.adv.pos_trajy[-1]) )
                else:
                    ego_t =  self.gl.adv.pos_trajx[-1]  / ego_speed if ego_speed > 0.1 else 0.1
                    bv_t = self.gl.adv.pos_trajy[-1] / adv_speed if adv_speed > 0.1 else 0.1
                    ped_obs.append( bv_t - ego_t )
                    ped_obs.append( abs(self.gl.adv.pos_trajy[-1]) )
                    if abs(adv_p[-1]) > 15:
                        self.gl.adv.pos_trajx.clear()
                        self.gl.adv.pos_trajy.clear()
            # 直接根据主车车辆坐标系 x与y来进行确认交互点
            elif len(self.gl.adv.pos_trajx) >= 1:
                ego_t =  abs(self.gl.adv.pos_trajx[-1])  / ego_speed if ego_speed > 0.1 else 0.1
                bv_t = abs(self.gl.adv.pos_trajy[-1]) / adv_speed if adv_speed > 0.1 else 0.1
                t =  bv_t - ego_t if abs(bv_t - ego_t) < 4 else 4
                ped_obs.append(t  )
                ped_obs.append( abs(self.gl.adv.pos_trajy[-1]) + 1 )
                print("egospeed, advspeed, pos:",ego_speed, adv_speed, self.gl.adv.pos_trajx[-1], self.gl.adv.pos_trajy[-1])
                print("bv_t - ego_t :",bv_t, ego_t,  ped_obs[-2] )
                print("dis :",ped_obs[-1] )
                print('************************************')
            # y2 = ego_p[0] * ex**3 + ego_p[1] * ex**2 + ego_p[2]*ex + ego_p[3]

        if self.gl.ego is not None:
            dx = self.gl.adv.pos_x -  self.gl.ego.pos_x
            dy = self.gl.adv.pos_y - self.gl.ego.pos_y
            dvx = self.gl.adv.vx - self.gl.ego.vx
            dvy = self.gl.adv.vy - self.gl.ego.vy

            ego_hdg = self.gl.ego.pos_h
            ego_lane_hdg = self.gl.ego.pos_h  + self.gl.ego.hdg 
            if ego_hdg > 3.14:
                ego_hdg -= 6.28
                ego_lane_hdg -=  6.28
            #模型输入 [主车纵向速度，主车横向速度，主车航向角，主车车道偏移，主车所在道路航向角，主车是否可以左转 ， 主车是否可以右转]
            traj_obs += [ self.model_manager.normalize_obs(dx,'x'),self.model_manager.normalize_obs( dy,'y') ,\
                self.model_manager.normalize_obs( dvx,'vx'),self.model_manager.normalize_obs( dvy,'vy'),\
                ego_hdg,  self.gl.ego.lane_offset, ego_lane_hdg, 1, 1 ]
            ita_obs +=  [ self.model_manager.normalize_obs(dx,'x'),self.model_manager.normalize_obs( dy,'y') ,\
                self.model_manager.normalize_obs( dvx,'vx'),self.model_manager.normalize_obs( dvy,'vy'),\
                ego_hdg,  self.gl.ego.lane_offset, ego_lane_hdg, 1, 1 ]
            ped_obs += [ self.model_manager.normalize_obs(dx,'x'),self.model_manager.normalize_obs( dy,'y') ,\
                self.model_manager.normalize_obs( dvx,'vx'),self.model_manager.normalize_obs( dvy,'vy'),\
                ego_hdg,  self.gl.ego.lane_offset, ego_lane_hdg, 1, 1 ]
        # 凑够剩余维度
        for i in range(5):
            # if i < len(self.gl.objects):
            #     traj_obs += [ self.model_manager.normalize_obs( self.gl.objects[i].pos_x ,'x'), self.model_manager.normalize_obs(self.gl.objects[i].pos_y,'y'),\
            #         self.model_manager.normalize_obs(self.gl.objects[i].vx,'vx'), self.model_manager.normalize_obs(self.gl.objects[i].vy,'vy')  ]
            # else:
            traj_obs += [0,0,0,0]

        traj_obs =  np.array(traj_obs).astype(self.space().dtype)
        ita_obs = np.array(ita_obs).astype(self.space().dtype)
        # if len(ped_obs) == 18:
        #     print('hhh')
        ped_obs = np.array(ped_obs).astype(self.space().dtype)
        return torch.tensor(traj_obs).unsqueeze(0),torch.tensor(ita_obs).unsqueeze(0),torch.tensor(ped_obs).unsqueeze(0)
    
    def sample_action(self, state):
        
        with torch.no_grad():
            actions = self.value_net(state.to(self.type).to(self.device))
            actions_lat = self.value_net_lat(state.to(self.type).to(self.device))
            return actions, actions_lat
    def sample_actions(self,state,ita_state,ped_state):
        # 使用训练好的模型进行动作选择
        with torch.no_grad():
            if self.in_juction and len(ita_state[0]) == 18:
                # ita_obs = np.random.rand(18).astype(self.space().dtype)
                # ita_obs = [0.33333333333333326, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.17994232725959, 8.644660822675124, -0.6007389418069842, -0.22916666666666663, 0.6333333333333333, 0.0, 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0]
                # # tensor([[ 1.8169e-02, -2.6485e-04, -1.4320e-02,  1.9289e-01,  4.2375e-07, 0.0000e+00,  0.0000e+00, -1.4369e+01,  9.4258e+00, -2.7060e-01,1.6031e-01,  1.9341e-02, -2.1530e-02,  1.6249e+00, -3.6091e-01, 1.5708e+00,  1.0000e+00,  1.0000e+00]])
                # ita_obs = [-0.3333,  0.0000,  3.1416,  0.0000,  3.1416,  0.0000,  0.0000, -0.2007, 18.0629,  0.4156, -0.2396, -0.3333,  0.3000, -1.5708,  0.0000, -1.5708, 0.0000,  0.0000]
                # ita_obs = np.array(ita_obs).astype(self.space().dtype)
                # ita_obs = torch.tensor(ita_obs).unsqueeze(0)
                # print('ita_state:',ita_state)
                # print('len ita_state:',len(ita_state[0]))
                lon_action = self.model_manager.model.value_net_ita( ita_state )
            elif self.gl.adv.obj_type == 5 and len(ped_state[0]) == 18:
                lon_action = self.model_manager.model.value_net_ped( ped_state )
            else:
                lon_action = self.model_manager.model.value_net(state)
            # lon_action1 = self.model_manager.model.value_net(state)
            lat_action = self.model_manager.model.value_net_lat(state)
            lon_action = lon_action[0].numpy()
            lat_action = lat_action[0].numpy()
            # 从当前状态获取动作
            lon_action = np.argmax(lon_action, keepdims=True)[0]
            lat_action =np.argmax(lat_action, keepdims=True)[0]
            # print("lon_index:",lon_action)
            # print("lat_index:",lat_action)
            lon_action = self.args.env_config['LON_ACTIONS'][lon_action]
            lat_action = self.args.env_config['LAT_ACTIONS'][lat_action]
        return {'lon':lon_action, 'lat':lat_action}

    # 控制指令执行函数
    def contrl(self):
        # 初始化
        if self.gl.pre_adv_lane_id is None:
            self.gl.pre_adv_lane_id = self.gl.adv.lane_id
        self.contrl_adv()
        # 更新上当前帧 对抗车车道id，用于下一帧
        self.gl.pre_adv_lane_id = self.gl.adv.lane_id
        if self.lib.get_msg_num()  ==  0 and self.gl.adv is not None:
            self.lib.addPkg(   self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, self.gl.adv.id, 1)
        print(">>>>>>>>>>>>>>>>>>>>ctrl>>>>>>>>>>>>>>>>>>>>:",self.ctrl_signal)

    # 碰撞警告，且会将坐标系转换成对抗车辆坐标系
    def collision_warning(self):
        # threshold
        
        other_vecs = []
        other_vecs += self.gl.objects
        other_vecs.append(self.gl.ego)
        # 建立对抗车车辆坐标系
        # for i in other_vecs:
            # i.show()
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        if self.gl.adv is not None :
            for i in other_vecs:
                
                # (self, base_obj, position_flag = False,velocity_flag = False,acc_flag = False ,rotate = -1):
                i.trans_cood2(self.gl.adv,True,True,True,rotate=1)    
            # for i in other_vecs:
            #     i.show()
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
            
            other_vecs = sorted(other_vecs, key=lambda x: math.sqrt(x.pos_x**2 + x.pos_y**2) )

            num = 0
            threshold = 2
            # 获取对抗车前后左右车辆
            for obj in other_vecs:
                if  self.gl.front_vec_to_compete is None and  obj.pos_x  >= 0 and abs(obj.pos_y) <=  threshold:
                    self.gl.front_vec_to_compete = obj
                    num += 1
                if  self.gl.left_neib_front_vec_to_compete is None and  obj.pos_x  >= 0 and obj.pos_y >  threshold:
                    self.gl.left_neib_front_vec_to_compete = obj
                    num += 1
                if self.gl.right_neib_front_vec_to_compete is None and obj.pos_x >= 0 and obj.pos_y < -threshold:
                    self.gl.right_neib_front_vec_to_compete = obj
                    num += 1
                if self.gl.bake_vec_to_compete is None and obj.pos_x < 0 and abs(obj.pos_y) <= threshold:
                    self.gl.bake_vec_to_compete = obj
                    num += 1
                if self.gl.left_neib_bake_vec_to_compete is None and obj.pos_x < 0 and obj.pos_y > threshold:
                    self.gl.left_neib_bake_vec_to_compete = obj
                    num += 1
                if self.gl.right_neib_bake_vec_to_compete is None and obj.pos_x < 0 and obj.pos_y < -threshold:
                    self.gl.right_neib_bake_vec_to_compete = obj
                    num += 1
                if num >= 6:
                    break

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print("adv:",self.gl.adv.name)
        if self.gl.front_vec_to_compete is not None:
            print("front_vec_to_compete:",self.gl.front_vec_to_compete.name)
        # print("left_neib_front_vec_to_compete:",self.gl.left_neib_front_vec_to_compete.name)
        # print("right_neib_front_vec_to_compete:",self.gl.right_neib_front_vec_to_compete.name)
        if self.gl.bake_vec_to_compete is not None:
            print("bake_vec_to_compete:",self.gl.bake_vec_to_compete.name)
        # print("left_neib_bake_vec_to_compete:",self.gl.left_neib_bake_vec_to_compete.name)
        # print("right_neib_bake_vec_to_compete:",self.gl.right_neib_bake_vec_to_compete.name)
        # 碰撞预警
        if self.gl.front_vec_to_compete is not None:
            # self.gl.front_vec_to_compete.show()
            if self.gl.front_vec_to_compete.vx < 0:
                ttc =self.get_speed(self.gl.front_vec_to_compete.pos_x - 5, self.gl.front_vec_to_compete.pos_y)  /  self.get_speed(self.gl.front_vec_to_compete.vx, self.gl.front_vec_to_compete.vy)
                if ttc < 0.5:
                    print("collision warning!!!  ttc:",ttc)
                    return True
        # for i in other_vecs:
        #     i.show()
        return False
    # 急刹指令
    def stop(self):
        self.lib.clear()
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -4, 0, self.gl.adv.id  , 1)   
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
        # print("v0_p:",v0_p)
        v0_p = np.array(v0_p)

        # print("vec1.pos_h",vec1.pos_h)
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


        # print("v1_p:",v1_p)
        # v1_p  = [ [-0.9999999196153099, -3.0000000267948956],[-1.0000000803846893, 2.9999999732051026], [0.9999999196153099, 3.0000000267948956], [1.0000000803846893, -2.9999999732051026]]
        # print("v1_p:",v1_p)
        # for i in v1_p:
        #     temp = list(map(lambda x,y:x + y, vp_p0,i))
        #     temp = list(map(lambda x,y:x + y, vp_p0,i))
        #     i[0] = temp[0]
        #     i[1] = temp[1]
        v1_p = np.array(v1_p)
        # print("v0_p:",v0_p)
        # print("v1_p:",v1_p)
        poly1 = Polygon(v0_p).convex_hull
        poly2 = Polygon(v1_p).convex_hull 
        # print("poly:",poly1,poly2)
        union_poly = np.concatenate((v0_p,v1_p))   #合并两个box坐标，变为8*2
        # print("union_poly:",union_poly)
        if not poly1.intersects(poly2): #如果两四边形不相交
            # print("如果两四边形不相交")
            return False
        
        else:
            # inter_area = poly1.intersection(poly2).area   #相交面积
            print(">>>>>>>>>>已碰撞!!!>>>>>>>>>>>>>>>>")
            return True
            
        #     # print("inter_area",inter_area)
        #     #union_area = poly1.area + poly2.area - inter_area
        #     union_area = MultiPoint(union_poly).convex_hull.area
        #     # print("union_area:",union_area)
        #     if union_area == 0:
        #         iou= 0
        #     print("inter_area/union_area:",inter_area, union_area)
        #     iou=float(inter_area) / union_area
        #     print("iou:",iou)
        #     # gap    fix   
        #     if iou > 0.013:
        #         return True
        # return False
    def get_time(self,obj):
        time = -1
        s = self.get_sqrt(self.prepare.confrontation_position[0] - obj.pos_x, self.prepare.confrontation_position[1] - obj.pos_y) 
        v = self.get_sqrt(obj.vx,  obj.vy)
        # print("s:",s)
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
                time +=1
            elif i.direction_to_ego == 1: 
                time -=2
            else:
                time += 0.5
            if time < 0:
                continue
            print("time, ego_time:",time, ego_time)
            _ , in_juction = self.chek_in_juction( i.roadId, 0,i.lane_id)
            if  abs(s) < 5 and v > 0 :
                self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -4, 0, i.id, 1)

            elif self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == 'STOP':
                print("decrease acc!!!",i.id)
                if  self.get_sqrt( i.vx,  i.vy) > 0:
                    self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -3, 0, i.id, 1)
                else :
                    self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, i.id, 1)
            # if self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == '':
            elif  time > ego_time:
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~i.name:",i.name)
                self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 3, 0, i.id, 1)
            elif time <= ego_time:
                print("decrease acc!!!")
                self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -5, 0, i.id, 1)
    
    def trans_world_cood(self):
        # 转换为全局坐标系
        if self.gl.adv is not None:
            
            if self.gl.ego.simFrame %5 == 0:
                self.gl.adv.pos_trajx.append(self.gl.adv.pos_x)
                self.gl.adv.pos_trajy.append(self.gl.adv.pos_y)
                # self.gl.ego.pos_trajx.append(self.gl.ego.pos_x)
                # self.gl.ego.pos_trajy.append(self.gl.ego.pos_y)
            self.gl.adv.to_dict(self.gl.ego, True, True,True,name=1)
        index = 2
        for i in self.gl.objects:
            i.to_dict(self.gl.ego, True,True,True,name=index)
            index+=1
    # 主循环函数
    def run_loop(self,ctrl):  
        self.ctrl_signal =  ctrl
        print("adv model is ready!!!!")
        for i in range(1):
            # 单帧处理函数
            self.vtd_exec_atomic()
            # 对抗车控制函数
            self.vtd_func()
        run = False
        while run:
            # 选定对抗车
            start = time.time()
            self.vtd_exec_atomic()
            dur_time = time.time() - start
            print("exec_atomic time:",dur_time)
            # 全局坐标系
            self.trans_world_cood()
            # other npc
            # if self.gl.adv is None or len(self.gl.objects) > 0:
            if  len(self.gl.objects) > 0:
                self.vtd_func()
            # if 0:
            if self.gl.adv is not None:
                # 打开对抗车所有大灯，标记选定对抗车
                if self.gl.compete_time % 10 == 1:
                # if self.gl.compete_time  == 1:
                    self.gl.scp.vec_light(self.gl.adv.name, left=True, right=True)
                    
                # 如果横向动作预留时间即将用完，当前帧切换为自动驾驶模式
                if self.keep_time > 0 and  self.keep_time_index + 1 == self.keep_time:
                    self.keep_time_index += 1
                    # print(">>>>>>>>>>>>>11111>>>>>>>>>>>",self.keep_time_index, self.keep_time)
                    self.autopilot(self.gl.adv.id)
                else:
                    # 获取模型输入
                    state,ita_state,ped_state = self.get_dqn_state()
                    # 获取模型输出指令
                    self.ctrl_signal = self.sample_actions(state,ita_state,ped_state)
                    # 碰撞检测，如果有碰撞危险返回True, 此处会将所有目标转换为对抗车辆坐标系
                    if self.collision_warning() :
                        # 刹车
                        self.stop()
                    else:
                        # 控制指令
                        self.contrl()
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


