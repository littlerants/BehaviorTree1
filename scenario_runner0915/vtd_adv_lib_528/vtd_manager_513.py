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
from gym_sumo.algo.global_route_planner_vtd_xodr import GlobalRoutePlanner
from gym_sumo.road.road import Road, RoadNetwork
from gym_sumo.road.lane import LineType, PolyLaneFixedWidth

from  vtd_adv_lib.global1 import GLOBAL
from  vtd_adv_lib.model_namager import ModelManager
from  vtd_adv_lib.scenario import SCENARIO, PREPARE
from  vtd_adv_lib.object import OBJECT
from  vtd_adv_lib.utils import *
from  vtd_adv_lib.head import *
from  vtd_adv_lib.mobil import LATCHECK
import pandas as pd

import logging
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
class ADV_Manager:
    # 速度阶梯查询表，用来计算对抗车预期加速度，假设车辆行驶速度范围0-30
    DEFAULT_TARGET_SPEEDS = np.linspace(0, 30, 10)


    def __init__(self, open_vtd = False,vtdPort = 48190,sensorPort = 48195, plugins_path = 'data/lib/sendTrigger.so',test = False,args = None):

        self.open_vtd = open_vtd
        self.args = args
        self.lat_check = LATCHECK()

        if test:
            self.model_manager =  ModelManager(self.args ,self.args.model_path)
        if self.open_vtd:
            # prepare
            # 需加载xodr，获取道路信息，并结构化道路
            self.map = GlobalRoutePlanner(self.args.xodr_path)
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
            self.prepare = PREPARE(self.args.scenario_path)
            # create static bojects
            self.static_objects_config = {}
            self.static_objects = []
            # world cood
            self.create_static_objects(self.prepare)
            self.wall_far = 70
            self.model_type = None

            # ACTIONS =  {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"}
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
            self.safe_dis  = 4
            # 真实距离
            self.dis = 0
            ##############################################################################################
            ##############################################################################################
            ##############################################################################################
            # 加载c++库，用来给对抗车发送纵向/横向指令，源文件请转至 your_project_path/ca/RDBClientSample/sendtrigger.cpp
            self.ll = ctypes.cdll.LoadLibrary
            # print("ADV---os.getcwd() + plugins_path:",os.getcwd() + plugins_path)
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
            # self.tcp_server.connect(("127.0.0.1",sensorPort))
            self.tcp_server.connect(("localhost",sensorPort))

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
                                                line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)))

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

    # 初始化或更新目标对象
    def create_objs(self):
        # self.gl.objects_set 所要维护的目标列表
        # 若为0，则直接清空 self.gl.objects（用于存储所有目标对象）
        if len(self.gl.objects_set) == 0:
            self.gl.objects.clear()
        # update ego objects
        # self.gl.ego_state 存储主车相关信息
        # 获取主车lane_id
        tmp_lane_id = None
        if 'lane_id' in self.gl.ego_state:
            tmp_lane_id =  self.gl.ego_state['lane_id']
        # 主车离路保护分支
        else:
            print("ADV---ego is not on road!!!!!!!!!")
            self.gl.ego  = None
            return
        # 根据主车lane_id 获取主车左右两侧lane_id信息
        leftLaneId  = self.gl.fram_data['LaneInfo'][tmp_lane_id]['leftLaneId'] if tmp_lane_id in self.gl.fram_data['LaneInfo'] else 127
        rightLaneId = self.gl.fram_data['LaneInfo'][tmp_lane_id]['rightLaneId'] if tmp_lane_id in self.gl.fram_data['LaneInfo'] else 127
        # lane 对象，用于后期获取lane的中心线等信息
        lane = None
        section = 0
        if 'lane_id' in self.gl.ego_state and 'roadId'  in self.gl.ego_state :
            lane_index, self.in_juction = self.chek_in_juction( self.gl.ego_state['roadId'],section ,self.gl.ego_state['lane_id'] )
            if self.gl.ego is None or  self.gl.ego.lane is None  or  (lane_index !=  self.gl.ego.lane.lane_index) :
                lane_index = lane_index + (0,)
                try:
                    lane = self.road.network.get_lane(lane_index)
                except:
                    pass
        # 如果主车不为空，更新主车各个参数
        if self.gl.ego is not None:

            self.gl.ego.update(simTime=self.gl.fram_data['simTime'], simFrame=self.gl.fram_data['simFrame'],\
            pos_x= self.gl.ego_state['x'], pos_y=self.gl.ego_state['y'],pos_h = self.gl.ego_state['h'] ,\
            vx=self.gl.ego_state['vx'], vy=self.gl.ego_state['vy'],acc_x=self.gl.ego_state['acc_x'], hdg=self.gl.ego_state['hdg'],\
            l=self.gl.ego_state['l'], w=self.gl.ego_state['w'],acc_y=self.gl.ego_state['acc_y'],roadId=self.gl.ego_state['roadId'],
            obj_type=self.gl.ego_state["type"],lane_offset=self.gl.ego_state['laneoffset'],lane_id=tmp_lane_id, \
            leftLaneId = leftLaneId, rightLaneId = rightLaneId,\
            distToJunc= self.gl.ego_state['distToJunc'] if 'distToJunc' in self.gl.ego_state else 1000000,\
            lane=lane,\
            light_state=self.gl.ego_state["light_state"] if 'light_state' in self.gl.ego_state else None
            )
        # 否则初始化主车对象
        else:
            self.gl.ego = OBJECT(simTime=self.gl.fram_data['simTime'], simFrame=self.gl.fram_data['simFrame'],name=self.gl.ego_state["name"], \
            id=self.gl.ego_state['id'], pos_x= self.gl.ego_state['x'],off_x=self.gl.ego_state['off_x'], pos_y=self.gl.ego_state['y'],pos_h=self.gl.ego_state['h'],\
            hdg=self.gl.ego_state['hdg'],  vx=self.gl.ego_state['vx'], vy=self.gl.ego_state['vy'],roadId=self.gl.ego_state['roadId'],\
            acc_x=self.gl.ego_state['acc_x'], l=self.gl.ego_state['l'], w=self.gl.ego_state['w'],\
            acc_y=self.gl.ego_state['acc_y'],obj_type=self.gl.ego_state["type"],lane_offset=self.gl.ego_state['laneoffset'],\
            lane_id=tmp_lane_id, leftLaneId=leftLaneId,rightLaneId = rightLaneId,\
            distToJunc= self.gl.ego_state['distToJunc'] if 'distToJunc' in self.gl.ego_state else 1000000,\
            lane=lane
            )
        # 遍历self.gl.fram_data['Objects'] 中探测到的目标，并初始化或更新这些目标
        
        # print("self.gl.fram_data:",self.gl.fram_data)
        for i in self.gl.fram_data['Objects']:
            # Npc车辆离路保护分支
            tmp_lane_id =  i['lane_id'] if 'lane_id' in i else 0
            leftLaneId  = self.gl.fram_data['LaneInfo'][ i['lane_id'] ]['leftLaneId'] if tmp_lane_id != 0 and  i['roadId'] == self.gl.ego.roadId and i['lane_id'] in self.gl.fram_data['LaneInfo']  else 127
            rightLaneId = self.gl.fram_data['LaneInfo'][ i['lane_id'] ]['rightLaneId'] if tmp_lane_id != 0 and i['roadId'] == self.gl.ego.roadId and i['lane_id'] in self.gl.fram_data['LaneInfo'] else 127
            
            # 如果在维护列表中，更新状态，否则初始化对象
            if i['id'] in self.gl.objects_set:
                for j in self.gl.objects:
                    if i['id']== j.id:
                        j.update(simTime=self.gl.fram_data['simTime'], simFrame=self.gl.fram_data['simFrame'],\
                        pos_x= i['x'], pos_y=i['y'],pos_h = i['h'],hdg=i['hdg'] ,vx=i['vx'], vy=i['vy'],\
                        acc_x=i['acc_x'], l=i['l'], w=i['w'],acc_y=i['acc_y'],obj_type=i["type"],\
                        lane_offset=i['laneoffset'] if 'laneoffset' in i else 0,lane_id = tmp_lane_id ,roadId=i['roadId'] if 'roadId' in i else 0,\
                        leftLaneId = leftLaneId, rightLaneId = rightLaneId,\
                        distToJunc= self.gl.ego_state['distToJunc'] if 'distToJunc' in self.gl.ego_state else 1000000,static = False
                            )
            else:
                obj = OBJECT(simTime=self.gl.fram_data['simTime'], simFrame=self.gl.fram_data['simFrame'],name=i["name"], \
                id=i['id'], pos_x= i['x'], pos_y=i['y'],off_x=i['off_x'], pos_h=i['h'],hdg=i['hdg'],  vx=i['vx'], vy=i['vy'],acc_x=i['acc_x'], \
                l=i['l'], w=i['w'],acc_y=i['acc_y'],obj_type=i["type"],\
                lane_offset=i['laneoffset'] if 'laneoffset' in i else 0 ,lane_id= tmp_lane_id ,roadId=i['roadId']  if 'roadId' in i else 0,\
                leftLaneId = leftLaneId, rightLaneId = rightLaneId,\
                distToJunc= self.gl.ego_state['distToJunc'] if 'distToJunc' in self.gl.ego_state else 1000000,static = False
                    )
                obj.initial_state2ego = {"pos_x":obj.pos_x,"pos_y":obj.pos_y,"pos_h":obj.pos_h}
                self.gl.objects_set.append(obj.id)
                self.gl.objects+= [obj]



        if len(self.static_objects) > 0:
            for i in self.static_objects:
                dis2ego = self.get_sqrt(self.static_objects_config[i.name]['pos_x'] - self.gl.ego.pos_x,\
                                        self.static_objects_config[i.name]['pos_y'] - self.gl.ego.pos_y)
                # print("dis2ego:",dis2ego)
                if dis2ego < 200 and  i.id not in self.gl.objects_set:
                    # static_tmp = copy.deepcopy(i)
                    self.gl.objects.append(i)
                    self.gl.objects_set.append(i.id)
                i.update(pos_x=self.static_objects_config[i.name]['pos_x'],pos_y=self.static_objects_config[i.name]['pos_y'],\
                         vx=0,vy = 0,acc_x=0, acc_y=0,\
                         lane=self.static_objects_config[i.name]['lane'], lane_id=self.static_objects_config[i.name]['lane_id'], roadId=self.static_objects_config[i.name]['roadId'], \
                         pos_h=self.static_objects_config[i.name]['pos_h'],static = True
                         )
                i.trans_cood2(self.gl.ego, True, True, True, rotate=1)

        for i in self.gl.objects:
            # ego_vx ,ego_vy = trans2angle(self.gl.ego.vx,self.gl.ego.vy,self.gl.ego.pos_h)
            # if i.vx + ego_vx < 1:
            #     i.static = True
            # add lane object
            lane = None
            # 更新lane对象
            section = 0
            lane_index = self.map.road_id_to_edge[i.roadId][section][i.lane_id]  + (0,)
            if i.lane is None:
                try:
                    i.lane = self.road.network.get_lane(lane_index)
                except:
                    pass
            elif   (lane_index != i.lane.lane_index):
                try:
                    lane = self.road.network.get_lane(lane_index)
                except:
                    pass
                i.update(lane=lane)




      #  print("ADV--- after self.gl.objects_set:",self.gl.objects_set)
    # object解析   pkg = 9
    def handleRDBitemObjects(self,simFrame,simTime,dataPtr,flag,data):
        # # print("ADV---simFrme:",simFrame)
        # # print("ADV---simTime:",simTime)
        # # print("ADV---dataPtr:",dataPtr)

        # RDB_OBJECT_STATE_BASE_t
        # uint32_t            id;
        # uint8_t             category;
        # uint8_t             type;
        # uint16_t            visMask;
        # char                name[32];
        # RDB_GEOMETRY_t      geo;
        # RDB_COORD_t         pos;
        # uint32_t            parent;
        # uint16_t            cfgFlags;
        # int16_t             cfgModelId;
        # 4 1 1 2 32 6*float    3*double 3*float 1 1 2 4 2 2   =112
        # RDB_OBJECT_STATE_EXT_t
        # RDB_COORD_t         speed;
        # RDB_COORD_t         accel;
        # float               traveledDist;
        # uint32_t            spare[3];
        # ddd           fff B B H
        # ddd           fff B B H
        # f
        # I                                =  96
        # I B B h 32c ffffff ddd fff B B H I H h
        result = []
        item_base = struct.unpack('I2Bh32c6f3d3f2BHIHh',data[dataPtr:dataPtr + 112])
        # # print("ADV---handleRDBitemObjects:",item_base)
        s = ''
        # print("ADV---id:",item_base[0])
        # print("ADV---type:",item_base[2])
        # add pedestrian
        # print("ADV---item_base:",item_base)
        for i in item_base[4:4 + 32]:
            if i.decode() != '\00':
                s += i.decode()
        # print("ADV---name:",s)


        if item_base[2] >= 1 and  item_base[2] <= 5:
        # if item_base[2] >= 1 and   item_base[2] <=4:
            result += [item_base[0] ] # list(item_base[0])

            # print("ADV---geometry:",item_base[36:36 + 6])
            # print("ADV---position:",item_base[36 + 6:36 + 6 + 9])
            # id, x, y ,h
            result += ( [ item_base[36 + 6] ] + [item_base[36 + 7] ] + [item_base[36 + 9]])

            if flag:
                item_exa = struct.unpack('dddfffBBHdddfffBBHf3I',data[dataPtr+ 112:dataPtr+ 112 + 96])
                # # print("ADV---item_exa:",item_exa)
                # print("ADV---speed:",item_exa[0:9])
                # vx,vy,vh
                result += ([item_exa[0]]  + [item_exa[1]]  + [ item_exa[3]])
                # geometry l , w ,type
                result += [  item_base[36], item_base[36 + 1] ,item_base[2]   ]
                # print("ADV---acceleration:",item_exa[9:9+9])
                # acc_x, acc_y
                result += [ item_exa[9] , item_exa[10] ]
                # off_x
                result += [item_base[36+3]]
                # print("ADV---distance:",item_exa[-4])
            result.append(s)
            # self.gl.vecs_set.append( result[0] )
            # print("ADV---[id, x,y ,h, vx, vy ,vh, l , w ,type ,name]:",result)
            result1 = {"id":result[0], "x":result[1] ,"y":result[2], "h":result[3], "vx":result[4], "vy":result[5]
                ,"vh":result[6], "l":result[7], "w":result[8], "type":result[9],"acc_x":result[10], "acc_y":result[11],"off_x":result[12],  "name":result[-1]
            }
            # if result[0] == 1 and  self.gl.ego is not None:
            #     self.gl.ego.update(pos_x= result1['x'], pos_y=result1['y'],pos_h = result1['h'] ,vx=result1['vx'], vy=result1['vy'],acc_x=result1['acc_x'], l=result1['l'], w=result1['w'],acc_y=result1['acc_y'],obj_type=result1["type"])

            # print("ADV---result1:",result1)
            if result1["id"] == 1 and result1["name"] == 'Ego' :
                self.gl.ego_state.update(result1)
            else:
                self.gl.fram_data['Objects'] += [result1]
            # return result
            return True
        return False

    # 道路pkg解析
    def handleRDBitemRoadPos(self,simFrame,simTime,dataPtr,flag,data):

        # uint32_t         playerId;
        # uint16_t         roadId;
        # int8_t           laneId;
        # uint8_t          flags;
        # float            roadS;
        # float            roadT;
        # float            laneOffset;
        # float            hdgRel;
        # float            pitchRel;
        # float            rollRel;
        # uint8_t          roadType;
        # uint8_t          spare1;
        # uint16_t         spare2;
        # float            pathS;

        item_base = struct.unpack('IHbB6f2BHf',data[dataPtr:dataPtr + 40])
        # print("ADV---handleRDBitemRoadPos:",item_base)

        # if item_base[0] != 1:
        #   #  print("ADV---handleRDBitemRoadPos others_vehicle:",GLOBAL.others_vehicle)
        #     GLOBAL.others_vehicle.append( [item_base[0],item_base[2]] )

        result1 = {"id":item_base[0] ,"roadId":item_base[1],"lane_id":item_base[2], "laneoffset":item_base[6], "hdg":item_base[7]}

        # if result1['id'] == 1 and  self.gl.ego is not None:
        #     self.gl.ego.update(hdg=result1['hdg'],lane_id = result1['lane_id'],roadId=result1['roadId'])
        if result1['id'] == 1:
            self.gl.ego_state.update(result1)
        else:
            self.gl.fram_data['RoadPos'] += [result1]
    # lane 消息解析
    def handleRDBitemLaneInfo(self,simFrame,simTime,dataPtr,flag,data ):
        # uint16_t    roadId;
        # int8_t      id;
        # uint8_t     neighborMask;
        # int8_t      leftLaneId;
        # int8_t      rightLaneId;
        # uint8_t     borderType;
        # uint8_t     material;
        # uint16_t    status;
        # uint16_t    type;
        # float       width;
        # double      curvVert;
        # double      curvVertDot;
        # double      curvHor;
        # double      curvHorDot;
        # uint32_t    playerId;
        # uint32_t    spare1;
        # 2 1 1 1 1 1 1 2 2 4 8 8 8 8 4 4            2+ 6*1 +  2*2  + 4 +  4*8 +2*4 = 56
        # H b B b b B B H H f d d d d I I               HbB2b2B2Hf4d2I
        item_base = struct.unpack('HbB2b2B2Hf4d2I',data[dataPtr:dataPtr + 56])
        # print("ADV---handleRDBitemLaneInfo:",item_base)
        # print("ADV---handleRDBitemLaneInfo ego_lane_id:",ego_lane_id)
        result1 = {"id":item_base[-2], "road_lane_id":item_base[1], "lanewidth":item_base[9], "leftLaneId":item_base[3], "rightLaneId":item_base[4] }
        self.gl.fram_data['LaneInfo'].update({item_base[1]:result1})
        return True
    # road pkg 解析
    def handleRDBitemROADSTATE(self,simFrame,simTime,dataPtr,flag,data):
        # C++ struct
        #  typedef struct
        #  {
        #      uint32_t     playerId;
        #      int8_t       wheelId;
        #      uint8_t      spare0;
        #      uint16_t     spare1;
        #      uint32_t     roadId;
        #      float        defaultSpeed;
        #      float        waterLevel;
        #      uint32_t     eventMask;
        #      float        distToJunc;
        #      int32_t      spare2[11];
        #  } RDB_ROAD_STATE_t;
        # 4  1  1  2  4  f  f  4 f 4*11
        # I  b  B  H  I  f  f  I f  11i

        item_base = struct.unpack('IbBHI2fIf11i',data[dataPtr:dataPtr + 72])
        result1 = {"id":item_base[0], "distToJunc":item_base[8],"roadId":item_base[4]}

        if result1['id'] == 1:
            self.gl.ego_state.update(result1)

        self.gl.fram_data["RoadState"] += [result1]
        return True
    # 交通标识解析
    def handleRDBitemTRAFFICLIGHT(self,simFrame,simTime,dataPtr,flag,data):
        #  typedef struct
        #  {
        #      RDB_TRAFFIC_LIGHT_BASE_t base;
        #      RDB_TRAFFIC_LIGHT_EXT_t  ext;
        #  } RDB_TRAFFIC_LIGHT_t;

        #  typedef struct
        #  {
        #     int32_t                   id;
        #     float                     state;
        #     uint32_t                  stateMask;
        #  } RDB_TRAFFIC_LIGHT_BASE_t;

        #  typedef struct
        #  {
        #     int32_t                   ctrlId;
        #     float                     cycleTime;
        #     uint16_t                  noPhases;
        #     uint32_t                  dataSize;
        #  } RDB_TRAFFIC_LIGHT_EXT_t;

        #  typedef struct
        #  {
        #     float   duration;
        #     uint8_t type;
        #     uint8_t spare[3];
        #  } RDB_TRAFFIC_LIGHT_PHASE_t;

        # i f I  i f H I  f B B*3
        # 4 4 4  4 4 2 4  4 1 1*3

        #define RDB_TRLIGHT_PHASE_OFF           0
        #define RDB_TRLIGHT_PHASE_STOP          1
        #define RDB_TRLIGHT_PHASE_STOP_ATTN     2
        #define RDB_TRLIGHT_PHASE_GO            3
        #define RDB_TRLIGHT_PHASE_GO_EXCL       4
        #define RDB_TRLIGHT_PHASE_ATTN          5
        #define RDB_TRLIGHT_PHASE_BLINK         6
        #define RDB_TRLIGHT_PHASE_UNKNOWN       7

        item_base = struct.unpack('ifI',data[dataPtr:dataPtr + 12])
        light_cur_state = item_base[1]
        # print("ADV---handleRDBitemTRAFFICLIGHT:",item_base)
        # print("ADV---flag:",flag)
        if flag:
            item_base1 = struct.unpack('ifHI',data[dataPtr + 12:dataPtr + 12 + 16])
            # print("ADV---handleRDBitemTRAFFICLIGHTexe:",item_base1)
        noPhases = item_base1[2]
        if noPhases >= 3:
            dataSize = item_base1[3]
            # print("ADV---dataSize:",dataSize)
            # zi jie dui qi
            phasePtr = dataPtr + 12 + 16
            light_stat = []
            for  i in range(noPhases):
                phase = struct.unpack('f4B',data[phasePtr:phasePtr +  8])
                # print("ADV---phase:",phase)
                light_stat += [ {"duration":phase[0], "type":phase[1]} ]
                phasePtr =  phasePtr + 8
            # print("ADV---light_cur_state:",light_cur_state)
            # print("ADV---light_stat:",light_stat)
            temp = light_cur_state
            while True:
                for i in light_stat:
                    temp -= i['duration']
                    if temp < 0:
                        if self.gl.close_light:
                            result1 = {"id":1,"ctrlId":item_base1[0],"light_cur_state":light_cur_state, "light_state":LIGHT_SIGNAL[ i['type'] ]}
                            # GLOBAL.fram_data['TRAFFIC_LIGHT'] += [result1]
                            self.gl.ego_state.update(result1)
                            self.gl.close_light = False
                        return True
        return False


    def parseRDBMessageEntry(self,simFrame, simTime, entry, entry_handle, data, ego_lane_id = None):
        # print("ADV---^^^^^^^^^^^^^^^^^^^^^parseRDBMessageEntry^^^^^^^^^^^^^^^^^^^^^")
        # # print("ADV---ego_lane_id:",ego_lane_id)
        # # print("ADV---entry:",entry)
        noElements = int( entry_handle[1] / entry_handle[2] ) if entry_handle[2] else 0
        # # print("ADV---noElements:",noElements)

        if noElements == 0:
            if entry_handle[-2] == RDB_PKG_ID_START_OF_FRAME:
               print(  "void parseRDBMessageEntry: got start of frame\n" )
            if entry_handle[-2] == RDB_PKG_ID_END_OF_FRAME:
               print(  "void parseRDBMessageEntry: got end of frame\n" )


        dataPtr = entry + entry_handle[0]
        vechile_state = False
        # print("ADV---pkg:",entry_handle[-2]
        # print("ADV---pkg/noElements:",entry_handle[-2],noElements)
        if entry_handle[-2] == RDB_PKG_ID_ROAD_POS:
            while noElements > 0:
                noElements -= 1
                vechile_state = self.handleRDBitemRoadPos(simFrame,simTime,dataPtr,entry_handle[-1],data)
                dataPtr += entry_handle[2]
        elif  entry_handle[-2] == RDB_PKG_ID_LANE_INFO:
            while noElements > 0:
                noElements -= 1
                vechile_state = self.handleRDBitemLaneInfo(simFrame,simTime,dataPtr,entry_handle[-1],data)
                dataPtr += entry_handle[2]
        elif entry_handle[-2] == RDB_PKG_ID_ROAD_STATE:
            while noElements > 0:
                noElements -= 1
                vechile_state = self.handleRDBitemROADSTATE(simFrame,simTime,dataPtr,entry_handle[-1],data)
                dataPtr += entry_handle[2]
            # if vechile_state is not None:
            #     return ["ego distToJunc",vechile_state]

        elif entry_handle[-2] == RDB_PKG_ID_OBJECT_STATE:
            # print("ADV---pkg == 9")
            while noElements > 0:
                noElements -= 1
                vechile_state = self.handleRDBitemObjects(simFrame,simTime,dataPtr,entry_handle[-1],data)
                dataPtr += entry_handle[2]


        elif  entry_handle[-2] == RDB_PKG_ID_TRAFFIC_LIGHT:

            while noElements > 0:
                noElements -= 1
                if not vechile_state:
                    # just need closest light sig
                    vechile_state = self.handleRDBitemTRAFFICLIGHT(simFrame,simTime,dataPtr,entry_handle[-1],data)

                dataPtr += entry_handle[2]


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
                if self.gl.adv and self.gl.adv.pos_x < -5:
                    self.gl.adv_flag = False
                return  True
            return False

    def parseRDBMessage(self,pData, data,handle):
        entry = pData + SIZE_RDB_MSG_HDR_t
        # # print("ADV---entry:",entry)

        remainingBytes = handle.dataSize
        # # # print("ADV---remainingBytes:",remainingBytes)
        # ego_trajectory = "<Query entity='player' name='Ego'><SteeringPath dt='0.05' noPoints='20'/></Query>".encode('utf-8')
        # handle = struct.pack('HH64s64sI{}s'.format(len(ego_trajectory)),40108,0x0001,"ExampleConsole".ljust(64,'\0').encode('utf-8'),"any".ljust(64,'\0').encode('utf-8') ,len( ego_trajectory ),ego_trajectory)
        # tcp_server_inf.send(handle)
        # 循环执行完毕，将获取到VTD仿真世界中当前帧所有信息
        while ( remainingBytes ):
            #
            # # print("ADV---^^^^^^^^^^^^^^^^^^^^^parseRDBMessage^^^^^^^^^^^^^^^^^^^^^:",remainingBytes)
            # # print("ADV---while entry:",entry)
            #  print("ADV---remainingBytes:",remainingBytes)
            try:
                entry_handle = struct.unpack('IIIHH',data[entry:entry + 16])
            except:
                break
            # 需要解析的pkg列表
            exec_pkg = [1,2,5,6,9,21,27]
            if entry_handle[3] in exec_pkg:
                result = self.parseRDBMessageEntry( handle.frameNo, handle.simTime, entry, entry_handle, data)
            #  print("ADV---entry pkg*:",entry_handle[-2])

            remainingBytes -= entry_handle[0] + entry_handle[1]
            if remainingBytes:
                entry = entry + entry_handle[0] + entry_handle[1]

        #  数据融合
        self.gl.fusion_rdb_data()
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

        show_log = 0
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
                self.gl.adv =copy.deepcopy( self.gl.objects[flag['index']])
        elif  front_close_vec_index != -1:
            self.gl.compete_time = 0
            pop_index.append(front_close_vec_index)
            self.gl.adv = copy.deepcopy(self.gl.objects[front_close_vec_index])
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
        return entry

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
        # print("ADV---target_speed: ", target_speed)
        # print("ADV---speed_index: ", speed_index)
        # 获取 对抗车与主车的距离

        # 速度较大，加速度会添加偏置var
        # if speed_index > 3:
        #     var = 1 - speed_index / 15
        #     acceleration = (acceleration + var) if acceleration > 0 else acceleration - var
        # 平滑处理
        # self.listdp.append(acceleration)
        # if len(self.listdp)  == 13:
        #     alpha = 0.2
        #     temp = np.array(self.listdp)
        #     temp = temp.astype(np.float)
        #     tempf = signal.filtfilt(self.B, self.A, temp)
        #
        #     ema = [tempf[0]]  # 初始EMA值等于第一个数据点
        #     for i in range(1, len(tempf)):
        #         ema.append(alpha * tempf[i] + (1 - alpha) * ema[-1])
        #   #  print(ema[-1])
        #     acceleration = tempf[-1]

        # 获取加速度最大最小值
        # max_acc = self.get_speed(self.gl.adv.vx + self.gl.ego.vx,self.gl.adv.vy + self.gl.ego.vy) / 10
        # min_acc = self.get_max_acc_new(adv_speed)
        # if max_acc < 5:
        # max_acc = 5
        # 在变道等动作前半段时，抑制减速度
        # if self.keep_time_index > 0 and self.keep_time_index < (self.keep_time>>1):
        #     max_acc = 4
        #     min_acc = -3

        # 加速度正负反转状态
        # acceleration = self.opt_conver_acc(acceleration)
        # 添加加速pkg
        # print("ADV---max_acc:",max_acc)
        # if acceleration > max_acc:
        #     acceleration = max_acc
        # if acceleration < min_acc:
        #         acceleration = min_acc
        # if acceleration > 0:
        #     acceleration += 0.5
        # else:
        #     acceleration -= 0.5
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
        if self.model_type in [1,2]:
            front_dis= -8
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
        if self.model_type == 0 and ahw and  ahw.pos_x < 35:
            safe_dis  = safe_dis * 0.4
            print("new safe dis:",safe_dis)
        left_lane = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
        right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
        if (ahw  and not ahw.static and ahw.pos_x < 10 and self.gl.adv.lane_id ==  ahw.lane_id):

            if left_lane and  self.check_lane_change_action(self.gl.left_neib_front_vec_to_compete, self.gl.left_neib_bake_vec_to_compete):
                self.ctrl_signal['lat'] = "LANE_LEFT"
                return False
            elif right_lane and self.check_lane_change_action(self.gl.right_neib_front_vec_to_compete,self.gl.right_neib_bake_vec_to_compete):
                self.ctrl_signal['lat'] = "LANE_RIGHT"
                return False
        if self.gl.adv.direction_to_ego == 0:
            #  主车在对抗车右侧，禁止左方向移动
            if self.gl.ego.pos_y < -2:
                left_lane = 0
            #  主车在对抗车左侧，禁止右方向移动
            if self.gl.ego.pos_y > 2:
                right_lane = 0
            if self.ctrl_signal['lat'] == 'LANE_RIGHT' and self.gl.right_neib_bake_vec_to_compete and abs(self.gl.right_neib_bake_vec_to_compete.pos_x) < abs(self.gl.ego.pos_x) :
                right_lane = 0
                return True
            if self.ctrl_signal['lat'] == 'LANE_LEFT' and self.gl.left_neib_bake_vec_to_compete and  abs(self.gl.left_neib_bake_vec_to_compete.pos_x) < abs(self.gl.ego.pos_x):
                left_lane = 0
                return True
            # lane_off_set -= gain
            print("ADV---left_lane and right_lane:",left_lane,right_lane)
        return (self.gl.ego.pos_x > -safe_dis and abs(self.gl.adv.lane_id -  self.gl.ego.lane_id) <= 1 ) or (self.gl.ego.pos_x > 0  and  self.gl.adv.lane_id !=  self.gl.ego.lane_id)

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

        # 是否执行横向对抗， keep_time == -1 证明此时可以进行横向对抗动作
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
            # 如果是变道指令，并且预留时间已经用完，但是变道还未完成，则再增加10帧预留时间，预留时间上限为120帧,120帧后强制清零   
            if self.action_marking in ['LANE_RIGHT','LANE_LEFT'] and self.gl.adv_hdg_num <= 30 and self.keep_time < 120 :
                self.keep_time += 5
            else:
                self.stop_lat_act()
                return -1
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
        show_log = 0
        # ctrl     {"lat":"LANE_LEFT", "lon":FASTER}


        # while True:
        self.bMsgComplete = False
        self.gl.clear_fram_data()

            # while not bMsgComplete:
        while not self.bMsgComplete:
            # data = self.tcp_server.recv(204800)
            start_time  = time.time()
            data = self.tcp_server.recv(409600)
            # data = self.tcp_server.recv(1024000)
            print("tcp_server time:", (time.time() - start_time) * 1000)
            ret  = len(data)
          #  print("ADV---********************************3*********************************",ret)

            if ret <=0:
                # print("ADV---recv error!!! ret < 0!!!")
                return
            # # print("ADV---ret:",ret)
            # # print("ADV---bytesInBuffer + ret:",bytesInBuffer + ret)
            # # print("ADV---bufferSize:",bufferSize)

            pData = 0
            # if bytesInBuffer + ret > bufferSize:

            #     bufferSize = bytesInBuffer + ret
            # if ret!= msg[2] + msg[3]:
            #     ret = msg[2] + msg[3]
            # # print("ADV---ret:",ret)

            self.bytesInBuffer += ret
          #  print("ADV---bytesInBuffer:",self.bytesInBuffer)

            if  self.bytesInBuffer >=  SIZE_RDB_MSG_HDR_t:

                self.handle.update(struct.unpack('HHIIId',data[pData:pData + 24]) )


                if self.handle.magicNo != RDB_MAGIC_NO:
                    self.bytesInBuffer = 0

                while self.bytesInBuffer >= ( self.handle.headerSize + self.handle.dataSize):
                    # print("ADV---aaa",self.bytesInBuffer ,( self.handle.headerSize + self.handle.dataSize))
                    # # print("ADV---bytesInBuffer:",bytesInBuffer)
                    msgSize =  self.handle.headerSize + self.handle.dataSize
                    if show_log:
                       print("ADV---bytesInBuffer:",ret,self.bytesInBuffer,self.handle.headerSize,msgSize)

                    # # print("ADV---msgSize:",msgSize)
                    # isImage = False
                    # 48195
                    if self.bytesInBuffer == ( self.handle.headerSize + self.handle.dataSize):
                        self.gl.fram_data['simFrame'] = self.handle.frameNo
                        self.gl.fram_data['simTime'] = self.handle.simTime
                        print("ADV---every fram:",self.gl.fram_data['simFrame'],self.gl.fram_data['simTime'])
                        time1 = time.time()
                        self.parseRDBMessage(pData,data, self.handle)
                        print("parseRDBMessage time : ", (time.time() - time1) * 1000)
                    pData += msgSize
                    self.bytesInBuffer -= msgSize
                    self.bMsgComplete = True
                    # update handle
                    if self.bytesInBuffer > self.handle.headerSize :
                        try:
                            self.handle.update(struct.unpack('HHIIId',data[pData:pData + 24]) )
                        except:
                            print("ADV---prare failed!!!")
                            continue

                    #     # inference time 10 ms
                # end_time = time.time()
                #  print('running times:',end_time - start_time)

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
        return spaces.Box(shape=(1, state_dim), low=-np.inf, high=np.inf,
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

    # 控制指令执行函数
    def contrl(self,lat_warn):
        # 初始化
        # if self.gl.pre_adv_lane_id is None:
        #     self.gl.pre_adv_lane_id = self.gl.adv.lane_id
        self.contrl_adv(lat_warn)
        # 更新上当前帧 对抗车车道id，用于下一帧
        # self.gl.pre_adv_lane_id = self.gl.adv.lane_id
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
        vx,vy = trans2angle(self.gl.adv.vx, self.gl.adv.vy, self.gl.adv.pos_h)
        self.safe_dis = (vx + self.gl.ego.vx)*0.1 + 6 + (0 if self.gl.ego.pos_x <0 and  self.gl.ego.vx <0  else math.pow(self.gl.ego.vx,2) / (2*(-self.get_max_acc_new(adv_speed) - 0.03* ((vx + self.gl.ego.vx)/10)   ) )   )
        self.safe_dis *= self.gl.dangerous
        self.dis = self.get_distance(self.gl.ego)
        left_lane,right_lane  = self.LRlane_change()
        adv_current_lon, _ = self.gl.ego.lane.local_coordinates(np.array([self.gl.adv.pos_x, self.gl.adv.pos_y]))
        # print("+++++++++++++++++++++++++++++++++++++++:",self.gl.adv.lane.length, adv_current_lon)
        distance_ratio = (30 - (self.gl.adv.lane.length - adv_current_lon)) / self.gl.adv.lane.length
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

        if self.gl.adv.direction_to_ego in [1, 2, 4, 5] and ehw:
            self.model_type = 1
            obs[0] -= (random.randint(1,self.gl.lon_offset) if obs[0] < 0 else 0)
            obs[9] =  -obs[0] - (self.safe_dis)
            obs.append(1.0)
        elif not ehw and  ahw and (ahw.pos_x < self.wall_far):
            offset = 5
            obs += [ahw.pos_x - offset, ahw.pos_y, ahw.vx, ahw.vy, ahw.pos_h]
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
                if (( obj.name.find("static") != -1 and  self.gl.front_vec_to_compete is None and obj.pos_x >= 0  and obj.lane_id == self.gl.adv.lane_id ) or
                        ( obj.name.find("static") == -1 and self.gl.front_vec_to_compete is None and obj.pos_x >= 0 and abs(obj.pos_y) <= threshold)):
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
            down_boundary = 5
            up_boundary = 6
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

    def run_loop(self,ctrl):
        self.ctrl_signal =  ctrl
        print("ADV---adv model is ready!!!!")
        for i in range(1):
            # 单帧处理函数
            self.vtd_exec_atomic()
            # 对抗车控制函数
            self.vtd_func()
        # while 1 :
        save_reference_time = False
        if save_reference_time:
            env_key = ["reference time"]
            df = pd.DataFrame(columns=env_key)
        while 1:
            print("ADV--- Time =================================================== ", time.time() * 1000)

            # for i in range(1000):
            loop_start_time = time.time()
            # 选定对抗车
            self.vtd_exec_atomic()
            print("vtd_exec_atomic time:", (time.time() - loop_start_time) * 1000)
            # top_process = subprocess.Popen('top -b -n 1 | head -n 20', stdout=subprocess.PIPE, shell=True)
            # output, _ = top_process.communicate()
            # print(output.decode())
            # self.ba_exec()
            # adv_enable
            if not self.check_adv_enable():
                self.ba_exec()
                continue
            if self.gl.adv and  not self.gl.adv_flag:
                print("======================adv process start too slow~!!!!! stop adv !!!!!=========================")
                self.stop(self.gl.adv.id)
                self.sendctrlmsg()
                continue
            # 全局坐标系
            self.trans_world_cood()
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
                    #
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
            if save_reference_time:
                data = np.array([[  time.time() - loop_start_time  ]])
                df1 = pd.DataFrame(data=data,columns=env_key)
                df = df.append(df1, ignore_index=True)
                # df = pd.concat(df,df1,ignore_index=True) df.append(df1, ignore_index=True)


            print("ADV---loop_total_time:", time.time() - loop_start_time)
        df.to_csv('./reference_time' + '.csv', encoding='utf_8_sig')
