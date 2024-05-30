import socket
import struct
import math
from networkx.generators import ego
from numpy import conjugate, string_, test
from numpy.core.multiarray import result_type
import os
import ctypes
import xml.etree.ElementTree as ET
import time
from collections import deque
import scipy.signal as signal
import numpy as np
from Utils.torch import to_device
import torch
import Utils
from gym_sumo import utils
from gymnasium import spaces
from models.dqn_net import Net, DuelingDQN

from shapely.geometry import Polygon,MultiPoint  #多边形
from argparse import ArgumentParser

# from torch._C import T
RDB_PKG_ID_START_OF_FRAME     =     1    # /**< sent as first package of a simulation frame                      @version 0x0100 */
RDB_PKG_ID_END_OF_FRAME       =     2  
RDB_PKG_ID_ROAD_POS           =     5  
RDB_PKG_ID_LANE_INFO          =     6
RDB_PKG_ID_OBJECT_STATE       =     9  
RDB_PKG_ID_ROAD_STATE         =    21    # /**< road state information for a given player                        @version 0x0100 */
RDB_PKG_ID_TRAFFIC_LIGHT      =    27    # /**< information about a traffic lights and their states              @version 0x0100 */

LIGHT_SIGNAL = {1:'STOP', 3:'GO',5:'ATT'}

RDB_MAGIC_NO  = 35712
SIZE_RDB_MSG_HDR_t = 24
show_log = 0


def rotate_operate(x,y,theta):
    tmpx = x
    tmpy = y

    x=  tmpx*math.cos(theta) - tmpy*math.sin(theta)
    y=  tmpy*math.cos(theta) + tmpx*math.sin(theta)   
    return [x,y]
def rotate_operate1(self,x,y,theta):
    tmpx = x
    tmpy = y

    x=  tmpx*math.cos(theta) + tmpy*math.sin(theta)
    y=  tmpy*math.cos(theta) - tmpx*math.sin(theta)   
    return [x,y]

class SCENARIO():
    def __init__(self, data_path  =  None):
        if data_path is None:
            print("data_path is None ,default to read '/data/' path file(xodr and xml)!!!")
            data_path = '/data/'
            for i in os.listdir(data_path):
                if i[-4:] == 'xodr':
                    # print(i)
                    xodr_path = os.path.join(data_path, i)
                if i[-3:] == 'xml':
                    # print(i)
                    scenario_path = os.path.join(data_path, i)
        else:
            flag = 5
            
            # t intersection
            if flag == 1:
                xodr_path     = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/dingzilukou/dingzilukou.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/dingzilukou/dingzilukou.xml'
            # zadao
            if flag == 2:
                xodr_path     = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/1027_01.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/1027_01.xml'
            # intersection
            if flag == 3:
               
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/intersection/stopandgo+.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/intersection/stopandgo+.xml'
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/test/stopandgo+.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/test/stopandgo+.xml'
                # '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/intersection'
            # roundabout
            if flag == 4:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/roundabout/roundabout/Roundabout_with_5_Three_Way_exits-0001.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/roundabout/roundabout/Roundabout_with_5_Three_Way_exits-0001.xml'
            # lanechange
            if flag == 5:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/3_lane_to_2_Lane-0001.xodr'
                scenario_path ='/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/3_lane_change_2_lane.xml'
            if flag == 6:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/2_lane_to_1_Lane-0001.xodr'
                scenario_path ='/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/2_lane_change_to_1_lane.xml'     
        
        self.scenario_path = scenario_path
        self.xodr_path = xodr_path

    def get_scenario_init(self,flag = 0):
        # 匝道
        pars = []
        if flag == 0:
            # ego_x,ego_y,h, adv_x,adv_y, h
            pars.append([1.73, -288.5,90,  5.6, -273.5, 90])
            pars.append( [5.5, -289,  90,  9.45, -274, 90 ])
            pars.append( [13.9, -54.66, 84.4, 16.4, -39.6, 75.5 ])
            pars.append(  [40.76, 16.27, 53.14, 50.1, 28.0, 47.7])
            pars.append(  [108.8, 64.6, 21.7, 121.91, 72.84, 12.7])
            pars.append(  [172.2, 77.8, 0,  187.33, 81.8, 0] )
        return pars





class OBJECT():
    def __init__(self,simTime = -1,simFrame = -1, name='',id=100,lane_id=0,pos_x=0,off_x=1.39,pos_y=0,pos_h=0,hdg=0,vx=0,vy=0,
                v_h=0,w=0,l=0,lane_offset=0,inertial_heading=0,lane_w=0,obj_type = '1',acc_x  = 0, acc_y = 0,
                leftLaneId = -1,rightLaneId = -3,  distToJunc=10000000000.0,light_state='GO',roadId=None,adv_vec=False,lane = None):
        self.simTime = simTime
        self.simFrame = simFrame
        self.name = name
        self.id = id
        self.lane_id =lane_id
        self.pos_x = pos_x
        self.off_x = off_x
        self.pos_y = pos_y
        self.pos_h = pos_h
        self.vx = vx
        self.vy = vy
        self.v_h = v_h
        self.hdg =hdg
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.w = w
        self.l = l
        self.lane_offset = lane_offset
        self.inertial_heading = inertial_heading
        self.lane_w = lane_w
        self.distToJunc = distToJunc
        self.obj_type = obj_type
        self.light_state = light_state
        self.leftLaneId = leftLaneId
        self.rightLaneId = rightLaneId
        self.roadId = roadId
        self.predisToconfrontation_position = 99999999
        self.new_object = True
        self.adv_vec = adv_vec
        self.lane = lane

    # rotate      -1为顺时针方向
                # 1 为逆时针方向
    def trans_cood1(self, base_obj, position_flag = False,velocity_flag = False,acc_flag = False ,rotate = -1):
        # x*math.cos(t) + y*math.sin(t)
        # global ego_lane_heading 
        # print("trans_cood ego_lane_heading")
        # print("self.id",self.id,self.name)
        theta =rotate * base_obj.pos_h
        # print("theta:",theta)
        
        # if position_flag:
        #     obj.pos_x =  pos_x*math.cos(theta) + pos_y*math.sin(theta)
        #     obj.pos_y = -(pos_y*math.cos(theta) - pos_x*math.sin(theta)) 
        if position_flag:
            pos_x = self.pos_x 
            pos_y = self.pos_y
            self.pos_x =  pos_x*math.cos(theta) + pos_y*math.sin(theta) + base_obj.pos_x
            self.pos_y =  pos_y*math.cos(theta) - pos_x*math.sin(theta) + base_obj.pos_y
            # if heading:
            self.pos_h += base_obj.pos_h

        if velocity_flag:
            vx = self.vx
            vy = self.vy
            self.vx =  vx*math.cos(theta) + vy*math.sin(theta) + base_obj.vx
            self.vy =  vy*math.cos(theta) - vx*math.sin(theta)   + base_obj.vy
        if acc_flag:
            acc_x = self.acc_x
            acc_y = self.acc_y
            # print("org acc:",self.acc_x,self.acc_y)
            self.acc_x =  acc_x*math.cos(theta) + acc_y*math.sin(theta) + base_obj.acc_x
            self.acc_y =  acc_y*math.cos(theta) - acc_x*math.sin(theta)  + base_obj.acc_y
        # rotate      -1为顺时针方向
                      # 1 为逆时针方向
    def trans_cood2(self, base_obj, position_flag = False,velocity_flag = False,acc_flag = False ,rotate = -1):

        theta =rotate * base_obj.pos_h

        if position_flag:
            pos_x = self.pos_x - base_obj.pos_x
            pos_y = self.pos_y - base_obj.pos_y
            self.pos_x =  pos_x*math.cos(theta) + pos_y*math.sin(theta) 
            self.pos_y =  pos_y*math.cos(theta) - pos_x*math.sin(theta) 
            # if heading:
            deta_heading = self.pos_h -  base_obj.pos_h 
            if deta_heading >= 0:
                self.pos_h = deta_heading if abs(deta_heading) < 3.1415926 else  deta_heading - 6.2831852 
            else:
                self.pos_h = deta_heading if abs(deta_heading) < 3.1415926 else  deta_heading + 6.2831852 
                
        if velocity_flag:
            vx = self.vx -  base_obj.vx
            vy = self.vy - base_obj.vy
            self.vx =  vx*math.cos(theta) + vy*math.sin(theta)
            self.vy =  vy*math.cos(theta) - vx*math.sin(theta)   
        if acc_flag:
            acc_x = self.acc_x - base_obj.acc_x
            acc_y = self.acc_y - base_obj.acc_y
            # print("org acc:",self.acc_x,self.acc_y)
            self.acc_x =  acc_x*math.cos(theta) + acc_y*math.sin(theta) 
            self.acc_y =  acc_y*math.cos(theta) - acc_x*math.sin(theta) 


            # print("trans acc:",self.acc_x,self.acc_y)
 
    def update(self,simTime=-1, simFrame=-1,lane_id=None,pos_x=None,off_x = None, pos_y=None,pos_h=None,hdg=None,vx=None,vy=None,
                v_h=None,w=None,l=None,lane_offset=None,inertial_heading=None,lane_w=None,obj_type = None,acc_x  = None, acc_y = None,
                leftLaneId = None,rightLaneId = None,  distToJunc=None,light_state=None,roadId=None,lane = None):
        if simTime != -1:
            self.simTime = simTime
        if simFrame != -1:
            self.simFrame = simFrame
        if lane_id != None:
            self.lane_id =lane_id
        if hdg != None:
            self.hdg =hdg
        if pos_x != None:
            self.pos_x = pos_x
        if off_x != None:
            self.off_x = off_x
        if pos_y != None:  
            self.pos_y = pos_y
        if pos_h!= None:
            self.pos_h = pos_h
        if vx!= None:
            self.vx = vx
        if vy != None:
            self.vy = vy
        if v_h != None:
            self.v_h = v_h
        if acc_x != None:
            self.acc_x = acc_x
        if acc_y != None:
            self.acc_y = acc_y
        if w != None:
            self.w = w
        if l != None:
            self.l = l
        if lane_offset != None:
            self.lane_offset = lane_offset
        if inertial_heading != None:
            self.inertial_heading = inertial_heading
        if lane_w != None:
            self.lane_w = lane_w
        if distToJunc != None:
            self.distToJunc = distToJunc
        if obj_type != None:
            self.obj_type = obj_type
        if light_state != None:
            self.light_state = light_state
        if leftLaneId != None:
            self.leftLaneId = leftLaneId
        if rightLaneId != None:
            self.rightLaneId = rightLaneId
        if roadId != None:
            self.roadId = roadId
        if lane != None:
            self.lane = lane



    def show(self):
       print("simTime:",self.simTime,"simFrame:",self.simFrame,"name:",self.name,"id:",self.id,"x/y:",self.pos_x, self.pos_y, "vx/vy:",self.vx,self.vy, "h:",self.pos_h ,"acc_x/y:",self.acc_x,self.acc_y,"roadId:",self.roadId, "lane_id:",self.lane_id )  
    
    def to_dict(self,ego = None, position_flag = False, velocity_flag = False,acc_flag = False, name = None):
        if ego is not  None: 
            self.trans_cood1(ego,position_flag,velocity_flag,acc_flag )
        
        if name is None:
            return {self.name:{"simTime":self.simTime,"simFrame":self.simFrame,"name":self.name,"id":self.id, "pos_x":self.pos_x,"pos_y":self.pos_y, "pos_h":self.pos_h, "vx":self.vx, "vy":self.vy}}
        else:
            return {name:{"simTime":self.simTime,"simFrame":self.simFrame,"name":self.name, "id":self.id,"pos_x":self.pos_x,"pos_y":self.pos_y, "pos_h":self.pos_h, "vx":self.vx, "vy":self.vy}}
    def get_sqrt(self,x,y):
        return math.sqrt(x**2 + y**2)






class SCP():
    def __init__(self, tc_port =48190 ):

        tcp_server_inf = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 设置端口复用，使程序退出后端口马上释放
        tcp_server_inf.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

        tcp_server_inf.connect(("127.0.0.1",48179))

      #  print("connect!!!")
        self.tcp_server_inf = tcp_server_inf
        self.tc_port = tc_port
        # self.temp = 0
        self.ins1 = 'on'
        self.ins2 = 'on'


    def start_scenario(self,project_path ='/home/sda/upload/VTD.2021/VTD.2021.3/bin/../Data/Projects/Current',scenario_file ='/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/1027_01.xml'):
        self.stop()
      #  print("start....")
        data = "<SimCtrl><Project name='SampleProject' path='{}' /></SimCtrl>".format(project_path)

        handle = self.get_handle(data)
        self.send(handle)
        data = "<TaskControl><RDB client='false' enable='true' interface='eth0' portRx='{}' portTx='{}' portType='TCP' /></TaskControl>".format(self.tc_port,self.tc_port)

        handle = self.get_handle(data)
        self.send(handle)

        data = "<SimCtrl><UnloadSensors /><LoadScenario filename='{}' /><Init mode='operation' /></SimCtrl>".format(scenario_file)

        handle = self.get_handle(data)
        self.send(handle)
        
        # data = "<Reply entity='taskControl'><Init source='moduleMgr' /></Reply>"
        # handle = self.get_handle(data)
        # self.send(handle)

        # data = "<Query entity='taskControl'><Init source='moduleMgr' /></Query>"
        # handle = self.get_handle(data)
        # self.send(handle)

        # data = "<SimCtrl><LayoutFile filename='/home/sda/upload/VTD.2021/VTD.2021.3/Develop/Communication/AdvAgentV2-code_refactoring/examples/av2_model_test/2000-link.xodr' /></SimCtrl>"
        # handle = self.get_handle(data)
        # self.send(handle)

        # data = "<SimCtrl><InitDone source='moduleMgr' /></SimCtrl>"
        # handle = self.get_handle(data)
        # self.send(handle)

        data ="<SimCtrl><InitDone place='checkInitConfirmation' /></SimCtrl>"
        handle = self.get_handle(data)
        self.send(handle)
        data = "<SimCtrl><Project name='SampleProject' path='{}' /></SimCtrl>".format(project_path)
        handle = self.get_handle(data)
        self.send(handle)
        data ="<SimCtrl><UnloadSensors /><LoadScenario filename='{}' /><Start mode='operation' /></SimCtrl>".format(scenario_file)
        handle = self.get_handle(data)
        self.send(handle)

    def stop(self):
        data = "<SimCtrl><Stop /></SimCtrl>"
      #  print("stop.....")
        handle = self.get_handle(data)
        self.send(handle)
    def get_handle(self,data):

        return  struct.pack('HH64s64sI{}s'.format(len(data)),40108,0x0001,"ExampleConsole".ljust(64,'\0').encode('utf-8'),"any".ljust(64,'\0').encode('utf-8') ,len( data ),data.encode('utf-8'))

    def send(self,handle):
        self.tcp_server_inf.send(handle)


    def turn_right(self,actor,num = -1):
        data = "<Traffic><ActionLaneChange direction='{}' force='true' delayTime='0.0' approveByDriver='false' activateOnExit='false' driverApproveTime='0' actor='{}' time='3.0'/></Traffic>"
        data = data.format(num,actor)
        handle = self.get_handle(data)
        self.send(handle)

    def turn_left(self,actor,num = 1):
        data = "<Traffic><ActionLaneChange direction='{}' force='true' delayTime='0.0' approveByDriver='false' activateOnExit='false' driverApproveTime='0' actor='{}' time='3.0'/></Traffic>"
        data = data.format(num,actor)
        handle = self.get_handle(data)        
        self.send(handle)


    def dacc(self,actor, target_speed = 20,type = None):
        # set ped's speed if type == 5
        # logging.debug("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
        if type is not None and type == 5:
            # m/s -> km/h
            target_speed *= 3.6
            data = "<Traffic><ActionMotion speed='{}' actor='{}'  rate='1' force='true' delayTime='0'/></Traffic>"
        else:
            data  = "<Traffic><ActionSpeedChange rate='3' target='{}' force='true' delayTime='0.0' activateOnExit='false' pivot='' actor='{}'/></Traffic>"
        data = data.format(target_speed, actor)
        handle = self.get_handle(data)        
        self.send(handle)
    def auto(self,actor):
        self.off_light(actor)

        data = "<Traffic> <ActionAutonomous enable='true' force='true' delayTime='0.0' activateOnExit='false' actor='{}'/> </Traffic>"
        
        data = data.format(actor)
        # logging.debug("data:{}".format(data))
        handle = self.get_handle(data)        
        self.send(handle)
    def off_light(self,actor):
        data = "<Player name='{}'><Light type='indicator right' state='off'/> <Light type='indicator left' state='off'/></Player>"
        data = data.format(actor)
        # logging.debug("data:{}".format(data))
        handle = self.get_handle(data)        
        self.send(handle)
    def on_light(self,actor):

        # self.temp += 1
        # if self.temp >= 100:
        #     self.temp  =  0
        #     tmp = self.ins1
        #     self.ins1 = self.ins2
        #     self.ins2 = tmp
        data = "<Player name='{}'><Light type='indicator right' state='{}'/> <Light type='indicator left' state='{}'/></Player>"
        data = data.format(actor,self.ins1,self.ins2)
        # logging.debug("data:{}".format(data))
        handle = self.get_handle(data)        
        self.send(handle)

    def vec_light(self,actor, left, right):
        left = 'on' if left else 'off'
        right = 'on' if right else 'off'
        data = "<Player name='{}'><Light type='indicator right' state='{}'/> <Light type='indicator left' state='{}'/></Player>"
        data = data.format(actor,left,right)
        handle = self.get_handle(data)        
        self.send(handle)

    def setPosInertial(self,actor, x,y,hdeg,id = None):
        if id is None:
            data = "<Set entity='player' id='' name='{}'><PosInertial hDeg='{}' x='{}' y='{}'/></Set>".format(actor,hdeg,x,y)
        else:
            data = "<Set entity='player' id='{}' name=''><PosInertial hDeg='{}' x='{}' y='{}'/></Set>".format(id,hdeg,x,y)
        handle = self.get_handle(data)        
        self.send(handle)

    def setPosInertial1(self,data):
        msg = ''
        # id, x,y,h
        for i in data:
            msg += "<Set entity='player' id='{}' name=''><PosInertial hDeg='{}' x='{}' y='{}'/></Set>".format(i[0],i[3],i[1],i[2])

        handle = self.get_handle(msg)        
        self.send(handle)
    def Laneoffset(self,name,offset):
        data  =  "<Player name='{}'><LaneOffset absolute='{}' time='0' s='0'/></Player>".format(name,offset)
        handle = self.get_handle(data)        
        self.send(handle)

class GLOBAL():
    scp = SCP()
    # vecs_set = []
    objects_set = []
    objects = []
    result = {}
    ego = None
    adv = None
    # 对抗车辆前后，左侧前后，右侧前后车辆
    left_neib_front_vec_to_compete = None
    left_neib_bake_vec_to_compete = None
    right_neib_front_vec_to_compete = None
    right_neib_bake_vec_to_compete = None
    front_vec_to_compete = None
    bake_vec_to_compete = None 
    compete_time = 0
    compete_time_range = 500
    last_compete_name= ''
    last_compete_name_id = -1
    show_log = True
    close_light = True
    fram_data = {"simFrame":0, "simTime":0,
        "Objects":[],"RoadPos":[],"LaneInfo":{},
        "RoadState":[],"ROADMARK":[],"TrafficSign":[],"TRAFFIC_LIGHT":[]}
    ego_state = {}


    def fusion_rdb_data(self):
        for i in  self.fram_data["Objects"]:
            # print("name/id",i['name'],i['id'])
            if len(self.fram_data['RoadPos']): 
                for j in self.fram_data['RoadPos']:
                    if i['id'] == j['id']: 
                        i.update(j)
            if len(GLOBAL.fram_data['RoadState']): 
                for j2 in GLOBAL.fram_data['RoadState']:
                    if i['id'] == j2['id']:
                        i.update(j2)

    def clear_fram_data(self):
        self.fram_data["Objects"] = []
        self.fram_data["RoadPos"] = []
        self.fram_data["LaneInfo"] = {}
        self.fram_data["RoadState"] = []
        self.fram_data["ROADMARK"] = []
        self.fram_data["TrafficSign"] = []
        self.fram_data["TRAFFIC_LIGHT"] = []
        self.left_neib_front_vec_to_compete = None
        self.left_neib_bake_vec_to_compete = None
        self.right_neib_front_vec_to_compete = None
        self.right_neib_bake_vec_to_compete = None
        self.front_vec_to_compete = None
        self.bake_vec_to_compete = None 
        self.close_light = True
        # self.vecs_set = []

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


from gym_sumo.algo.global_route_planner_vtd_xodr import GlobalRoutePlanner
from gym_sumo.road.road import Road, RoadNetwork
from gym_sumo.road.lane import LineType, StraightLane, PolyLaneFixedWidth
from shapely.geometry import LineString, Point
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

class VTD_Manager:
    DEFAULT_TARGET_SPEEDS = np.linspace(0, 30, 15)
    def __init__(self, open_vtd = False,vtdPort = 48190,sensorPort = 48195, plugins_path = '/RDBClientSample/sendTrigger0109.so',test = False,args = None):

        self.open_vtd = open_vtd
        self.scenario = SCENARIO(1)
        args.env_config['map_file_path'] = self.scenario.xodr_path
        self.args = args

        if test:
            self.model_manager =  ModelManager(self.args ,self.args.output_path)
        if self.open_vtd:
            # prepare
            self.map = GlobalRoutePlanner(self.scenario.xodr_path)
            self.road = self.make_road(self.map)
            self.meeting_points = None
            self.junction_edge_list = set()
            self.junction_road_info = dict()
            self.juction_set()
            self.in_juction = False
            # self.close_light = True

            self.prepare = PREPARE(self.scenario.scenario_path)

            # ACTIONS =  {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"}
            self.ACTIONS_DUR =  { 'LANE_LEFT': 80, 'IDLE': 0 , 'LANE_RIGHT': 80, 'FASTER': 10, 'SLOWER': 10,
                                "LEFT_1": 50, "LEFT_2": 50, "RIGHT_1": 50, "RIGHT_2": 50}
            self.action_marking = ''         
            self.keep_time = -1
            self.keep_time_index = -1
            self.disappear_num = 0
            self.ctrl_signal = None
            self.safe_dis  = 4
            self.dis = 0

            self.ll = ctypes.cdll.LoadLibrary
            # print("os.getcwd() + plugins_path:",os.getcwd() + plugins_path)
            os_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/ca'
            # self.lib = self.ll(os.getcwd() + plugins_path)
            self.lib = self.ll(os_path + plugins_path)
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

            self.gl = GLOBAL()
            # self.gl.get_confrontation()


            self.ret = 0
            self.bytesInBuffer = 0
            self.pData = 0
            self.handle = HANDLE()

            self.listdp = deque(range(10),maxlen=10)
            # 
            
            self.N  = 3   # Filter order
            self.Wn = 0.1 # Cutoff frequencyself.B
            self.B = None
            self.A = None
            self.B, self.A = signal.butter(self.N, self.Wn, output='ba')
            # 检测加速度正负反转状态
            self.C_acc = []
            self.C_time = -1
            self.C = 0
# for i in range(100):
#     listdp.append(i)
#   #  print(listdp[0])

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
        print("11111:",roadId,section, lane_id)
        lane_index = self.map.road_id_to_edge[  roadId  ][section][ lane_id ]
        if lane_index in self.junction_edge_list:
            in_juction = True
        return lane_index,  in_juction






    def create_objs(self):

        if len(self.gl.objects_set) == 0:
            self.gl.objects.clear()
        # update ego objects
        tmp_lane_id =  self.gl.ego_state['lane_id']
        leftLaneId  = self.gl.fram_data['LaneInfo'][tmp_lane_id]['leftLaneId'] if tmp_lane_id in self.gl.fram_data['LaneInfo'] else 127
        rightLaneId = self.gl.fram_data['LaneInfo'][tmp_lane_id]['rightLaneId'] if tmp_lane_id in self.gl.fram_data['LaneInfo'] else 127
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

        for i in self.gl.fram_data['Objects']:

            tmp_lane_id = i['lane_id']
            leftLaneId  = self.gl.fram_data['LaneInfo'][ i['lane_id'] ]['leftLaneId'] if i['roadId'] == self.gl.ego.roadId and i['lane_id'] in self.gl.fram_data['LaneInfo']  else 127
            rightLaneId = self.gl.fram_data['LaneInfo'][ i['lane_id'] ]['rightLaneId'] if i['roadId'] == self.gl.ego.roadId and i['lane_id'] in self.gl.fram_data['LaneInfo'] else 127
            
            if i['id'] in self.gl.objects_set:
                for j in self.gl.objects:
                    if i['id']== j.id:
                        j.update(simTime=self.gl.fram_data['simTime'], simFrame=self.gl.fram_data['simFrame'],\
                        pos_x= i['x'], pos_y=i['y'],pos_h = i['h'],hdg=i['hdg'] ,vx=i['vx'], vy=i['vy'],\
                        acc_x=i['acc_x'], l=i['l'], w=i['w'],acc_y=i['acc_y'],obj_type=i["type"],\
                        lane_offset=i['laneoffset'],lane_id= tmp_lane_id,roadId=i['roadId'],\
                        leftLaneId = leftLaneId, rightLaneId = rightLaneId,\
                        distToJunc= self.gl.ego_state['distToJunc'] if 'distToJunc' in self.gl.ego_state else 1000000
                            )
            else:
                obj = OBJECT(simTime=self.gl.fram_data['simTime'], simFrame=self.gl.fram_data['simFrame'],name=i["name"], \
                id=i['id'], pos_x= i['x'], pos_y=i['y'],off_x=self.gl.ego_state['off_x'],pos_h=i['h'],hdg=i['hdg'],  vx=i['vx'], vy=i['vy'],acc_x=i['acc_x'], \
                l=i['l'], w=i['w'],acc_y=i['acc_y'],obj_type=i["type"],lane_offset=i['laneoffset'],lane_id= tmp_lane_id ,roadId=i['roadId'],\
                leftLaneId = leftLaneId, rightLaneId = rightLaneId,\
                distToJunc= self.gl.ego_state['distToJunc'] if 'distToJunc' in self.gl.ego_state else 1000000
                    )
                self.gl.objects_set.append(obj.id)
                self.gl.objects+= [obj]            
      #  print(" after self.gl.objects_set:",self.gl.objects_set)
    # object解析
    def handleRDBitemObjects(self,simFrame,simTime,dataPtr,flag,data):
        # # print("simFrme:",simFrame)
        # # print("simTime:",simTime)
        # # print("dataPtr:",dataPtr)
        
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
        # # print("handleRDBitemObjects:",item_base)
        s = ''
        # print("id:",item_base[0])
        # print("type:",item_base[2])
        # add pedestrian 
        # print("item_base:",item_base)
        for i in item_base[4:4 + 32]:
            if i.decode() != '\00':
                s += i.decode()
        # print("name:",s)


        if item_base[2] >= 1 and  item_base[2] <= 5:
        # if item_base[2] >= 1 and   item_base[2] <=4:
            result += [item_base[0] ] # list(item_base[0])

            # print("geometry:",item_base[36:36 + 6])
            # print("position:",item_base[36 + 6:36 + 6 + 9])
            # id, x, y ,h
            result += ( [ item_base[36 + 6] ] + [item_base[36 + 7] ] + [item_base[36 + 9]])
            
            if flag:
                item_exa = struct.unpack('dddfffBBHdddfffBBHf3I',data[dataPtr+ 112:dataPtr+ 112 + 96])
                # # print("item_exa:",item_exa)
                # print("speed:",item_exa[0:9])
                # vx,vy,vh
                result += ([item_exa[0]]  + [item_exa[1]]  + [ item_exa[3]])
                # geometry l , w ,type
                result += [  item_base[36], item_base[36 + 1] ,item_base[2]   ]
                # print("acceleration:",item_exa[9:9+9])
                # acc_x, acc_y
                result += [ item_exa[9] , item_exa[10] ]
                # off_x
                result += [item_base[36+3]]
                # print("distance:",item_exa[-4])
            result.append(s)
            # self.gl.vecs_set.append( result[0] )
            # print("[id, x,y ,h, vx, vy ,vh, l , w ,type ,name]:",result)
            result1 = {"id":result[0], "x":result[1] ,"y":result[2], "h":result[3], "vx":result[4], "vy":result[5]
                ,"vh":result[6], "l":result[7], "w":result[8], "type":result[9],"acc_x":result[10], "acc_y":result[11],"off_x":result[12],  "name":result[-1]
            }
            # if result[0] == 1 and  self.gl.ego is not None:
            #     self.gl.ego.update(pos_x= result1['x'], pos_y=result1['y'],pos_h = result1['h'] ,vx=result1['vx'], vy=result1['vy'],acc_x=result1['acc_x'], l=result1['l'], w=result1['w'],acc_y=result1['acc_y'],obj_type=result1["type"])

            # print("result1:",result1)
            if result[0] == 1:
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
        # print("handleRDBitemRoadPos:",item_base)

        # if item_base[0] != 1:
        #   #  print("handleRDBitemRoadPos others_vehicle:",GLOBAL.others_vehicle)
        #     GLOBAL.others_vehicle.append( [item_base[0],item_base[2]] )

        result1 = {"id":item_base[0] ,"roadId":item_base[1],"lane_id":item_base[2], "laneoffset":item_base[6], "hdg":item_base[7]}

        # if result1['id'] == 1 and  self.gl.ego is not None:
        #     self.gl.ego.update(hdg=result1['hdg'],lane_id = result1['lane_id'],roadId=result1['roadId'])
        if result1['id'] == 1:
            self.gl.ego_state.update(result1)
        else:
            self.gl.fram_data['RoadPos'] += [result1]

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
        # print("handleRDBitemLaneInfo:",item_base)
        # print("handleRDBitemLaneInfo ego_lane_id:",ego_lane_id)
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

        GLOBAL.fram_data["RoadState"] += [result1]
        return True

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
        # print("handleRDBitemTRAFFICLIGHT:",item_base)
        # print("flag:",flag)
        if flag:
            item_base1 = struct.unpack('ifHI',data[dataPtr + 12:dataPtr + 12 + 16])
            # print("handleRDBitemTRAFFICLIGHTexe:",item_base1)
        noPhases = item_base1[2]
        if noPhases >= 3:
            dataSize = item_base1[3]
            # print("dataSize:",dataSize)
            # zi jie dui qi
            phasePtr = dataPtr + 12 + 16
            light_stat = []
            for  i in range(noPhases):
                phase = struct.unpack('f4B',data[phasePtr:phasePtr +  8])
                # print("phase:",phase)
                light_stat += [ {"duration":phase[0], "type":phase[1]} ]
                phasePtr =  phasePtr + 8
            # print("light_cur_state:",light_cur_state)
            # print("light_stat:",light_stat)
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
        # print("^^^^^^^^^^^^^^^^^^^^^parseRDBMessageEntry^^^^^^^^^^^^^^^^^^^^^")
        # # print("ego_lane_id:",ego_lane_id)
        # # print("entry:",entry)
        noElements = int( entry_handle[1] / entry_handle[2] ) if entry_handle[2] else 0
        # # print("noElements:",noElements)

        if noElements == 0:
            if entry_handle[-2] == RDB_PKG_ID_START_OF_FRAME:
               print(  "void parseRDBMessageEntry: got start of frame\n" )
            if entry_handle[-2] == RDB_PKG_ID_END_OF_FRAME:
               print(  "void parseRDBMessageEntry: got end of frame\n" )


        dataPtr = entry + entry_handle[0]
        vechile_state = False
        # print("pkg:",entry_handle[-2]
        # print("pkg/noElements:",entry_handle[-2],noElements)
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
            # print("pkg == 9")
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
    def parseRDBMessage(self,pData, data,handle):
        entry = pData + SIZE_RDB_MSG_HDR_t
        # # print("entry:",entry)
        
        remainingBytes = handle.dataSize
        # # # print("remainingBytes:",remainingBytes)
        # ego_trajectory = "<Query entity='player' name='Ego'><SteeringPath dt='0.05' noPoints='20'/></Query>".encode('utf-8')
        # handle = struct.pack('HH64s64sI{}s'.format(len(ego_trajectory)),40108,0x0001,"ExampleConsole".ljust(64,'\0').encode('utf-8'),"any".ljust(64,'\0').encode('utf-8') ,len( ego_trajectory ),ego_trajectory)
        # tcp_server_inf.send(handle)
        while ( remainingBytes ):
            # 
            # # print("^^^^^^^^^^^^^^^^^^^^^parseRDBMessage^^^^^^^^^^^^^^^^^^^^^:",remainingBytes)
            # # print("while entry:",entry)
            #  print("remainingBytes:",remainingBytes)
            try:
                entry_handle = struct.unpack('IIIHH',data[entry:entry + 16])
            except:
                break
            #  print("entry_handle:",entry_handle)
            exec_pkg = [5,6,9,21,27]
            if entry_handle[3] in exec_pkg:
                result = self.parseRDBMessageEntry( handle.frameNo, handle.simTime, entry, entry_handle, data)
            #  print("entry pkg*:",entry_handle[-2])
            
            remainingBytes -= entry_handle[0] + entry_handle[1]
            if remainingBytes:
                entry = entry + entry_handle[0] + entry_handle[1]
        
        #  print("$$$$$$$$$$$$$$start1$$$$$$$$$$$$$$$")
        self.gl.fusion_rdb_data()
        #  print("$$$$$$$$$$$$$$start2$$$$$$$$$$$$$$$")
        self.create_objs()
        #      print("$$$$$$$$$$$$$$start3$$$$$$$$$$$$$$$")
        # for i in self.gl.objects:
        #         i.show()
        self.gl.objects = sorted(self.gl.objects, key=lambda x: math.sqrt(x.pos_x**2 + x.pos_y**2) )
        # print("$$$$$$$$$$$$$$end$$$$$$$$$$$$$$$")
        flag = {'index':0,"flg":False}
        pop_index = []
        index = 0
        front_close_vec_index = -1
        self.gl.adv = None
        show_log = 0
        if show_log:
           print("len(self.gl.objects):",len(self.gl.objects))
        if len(self.gl.objects) >0:
            # print("len objects:",len(self.gl.objects))g
            for i in self.gl.objects:
                # ego front vecs
                # print('distance:',self.get_sqrt(i.pos_x, i.pos_y))
                # 主车50米以内所有车辆
                if self.get_sqrt(i.pos_x, i.pos_y) <40:
                    # 主车后30米以内所有车辆
                    if  i.pos_x > -30:
                        # print("i.pos_x > 3")
                        # i.show()
                        # print("self.gl.compete_time:",self.gl.compete_time)
                        # 选定一辆对抗车后，对抗时间持续 500 fps
                        if self.gl.compete_time < self.gl.compete_time_range :
                            self.gl.compete_time += 1
                            if front_close_vec_index == -1:
                                front_close_vec_index = index
                            if i.name ==  self.gl.last_compete_name:
                                pop_index.append(index)
                                flag['flg'] = True
                                flag['index'] = index
                        else:
                            # auto pilot
                            self.gl.compete_time = 0
                            self.gl.last_compete_name = ''
                            self.gl.last_compete_name_id = -1
                          #  print("comete time is over!!!")
                            return None
                else:
                    pop_index.append(index)

                index += 1

        else:
            # print()
            self.gl.compete_time = 0
            print("ego near no compete vecs !!!")
            return None
        
        if show_log:
           print("flag::",flag)
           print("front_close_vec_index:",front_close_vec_index)
        if flag['flg'] :
            self.gl.adv = self.gl.objects[flag['index']]
        
 
        elif  front_close_vec_index != -1:
            self.gl.compete_time = 0
            pop_index.append(front_close_vec_index)
            self.gl.adv = self.gl.objects[front_close_vec_index]
            # self.gl.objects.pop(front_close_vec_index)
            # self.gl.objects_set.remove(self.gl.objects[front_close_vec_index].id)
        else:
          #  print("ego front no compete vecs !!!")
            self.gl.compete_time = 0
            return None
        # print("self.gl.adv.name:",self.gl.adv.name)
        # add lane object
        lane = None
        section = 0
        if self.gl.adv is not None:
            lane_index = self.map.road_id_to_edge[ self.gl.adv.roadId ][ section ][ self.gl.adv.lane_id ]
            if self.gl.adv.lane is  None  or  (lane_index !=  self.gl.adv.lane.lane_index) :
                lane_index = lane_index + (0,)
                try:
                    lane = self.road.network.get_lane(lane_index)
                except:
                    pass
                self.gl.adv.update(lane=lane)
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

        
        # 危险碰撞检测
        


        return entry
            

                # gl.scp.dacc(i.name, get_sqrt(i.vx,i.vy) - 3)
    def get_distance(self,ego):
        return math.sqrt( math.pow(ego.pos_x,2) + math.pow( ego.pos_y,2) )
    def get_speed(self,vx,vy):
        return math.sqrt( math.pow(vx,2) + math.pow( vy,2) )

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
        if speed < 5:
            return -5 
        elif speed > 20:
            return -3.5
        else:
            return -speed*0.1 - 5.5

    
    def lane_left_act(self):
        self.gl.scp.turn_left(self.gl.adv.name)

    def idle_act(self):
        pass
        # self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0.1, 0, self.gl.adv.id  , 1)
    def lane_right_act(self):
        self.gl.scp.turn_right(self.gl.adv.name)


    def faster_act(self,var = 1):
        
        adv_speed = self.get_speed(self.gl.adv.vx, self.gl.adv.vy)
        print("adv_speed:",adv_speed)
        print(self.gl.adv.vx,self.gl.adv.vy)

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

        acceleration = target_speed  - adv_speed

        print(">>>>act:::",adv_speed,target_speed,acceleration)
        self.dis = self.get_distance(self.gl.ego)


        if speed_index > 3:
            var = 1 - speed_index / 15
            acceleration = acceleration + var if acceleration > 0 else acceleration - var

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

       
        max_acc = self.get_speed(self.gl.adv.vx + self.gl.ego.vx,self.gl.adv.vy + self.gl.ego.vy) / 10 
        min_acc = self.get_max_acc(adv_speed)
        if max_acc < 3:
            max_acc = 3
        # 在变道等动作前半段时，抑制减速度
        if self.keep_time_index > 0 and self.keep_time_index < (self.keep_time>>1):
            max_acc = 3
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
        
        self.safe_dis = (self.gl.adv.vx + self.gl.ego.vx)*0.1 + 4 + math.pow(self.gl.ego.vx,2) / (2*(-self.get_max_acc(adv_speed) - 0.03* ((self.gl.adv.vx + self.gl.ego.vx)/10)   ) )
        if self.dis < self.safe_dis + 3 and self.ctrl_signal['lat'] in ['LANE_LEFT','LANE_RIGHT'] and self.gl.adv.lane_id != self.gl.ego.lane_id:
            acceleration = 1
        print("safe_dis:", (self.safe_dis ))
        print("dis:",self.dis)
        # # print("self.gl.ego.pos_x:",self.gl.ego.pos_x)
        # if self.gl.ego.pos_x < 0 and abs(self.gl.adv.vx + self.gl.ego.vx) < 10 and   dis < (safe_dis if safe_dis > 10 else 10):
        #     acceleration = 1
        # if self.in_juction:
        #     if adv_speed >= self.get_speed(self.gl.adv.vx + self.gl.ego.vx,self.gl.adv.vy + self.gl.ego.vy)* 2:
        #         acceleration = 0 
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], acceleration, 0, self.gl.adv.id  , 1)

    def opt_conver_acc(self,acceleration):
        print("self.C_time:",self.C_time)
        print("self.C_time:",self.C)
        print("acceleration:",acceleration)
        # 加速度正负反转状态
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
        
        print("self.C_acc:",self.C_acc)
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
        if self.dis < self.safe_dis:
            return
        left_lane  = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
        right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
        #  主车在对抗车右侧，禁止左方向移动
        if self.gl.ego.pos_y < 0:
            left_lane = 0
        if self.gl.ego.pos_y > 0:
            right_lane = 0
        # lane_off_set = abs(self.gl.ego.pos_y) - self.gl.ego.w/2 - self.gl.adv.w/2
        # lane_off_set = lane_off_set if lane_off_set > 1.8  else 2
        if left_lane or right_lane:
            lane_off_set = 1.6
        else:
            lane_off_set = 1.3
        # lane_off_set = 0.5
    
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

    def contrl_adv(self):
        # ACTIONS =  {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER',5: "LEFT_1", 6: "LEFT_2", 7: "RIGHT_1", 8: "RIGHT_2"}
        #     self.ACTIONS_DUR =  { 'LANE_LEFT': 40, 'IDLE': 0 , 'LANE_RIGHT': 40, 'FASTER': 10, 'SLOWER': 10,
        #                         "LEFT_1": 20, "LEFT_2": 20, "RIGHT_1": 20, "RIGHT_2": 20}
        # print("self.ctrl_signal:",self.ctrl_signal)

        if self.ctrl_signal['lon'] == 'FASTER':
            self.faster_act()
        elif self.ctrl_signal['lon'] == 'SLOWER':
            self.slower_act()
        elif self.ctrl_signal['lon'] == 'IDLE':
            self.idle_act()

        # self.ctrl_signal['lat'] = "RIGHT_2"
        if self.get_speed( self.gl.adv.vx, self.gl.adv.vy) <  15:
            return
        if self.ctrl_signal['lat'] == 'IDLE':
            self.action_marking = 'IDLE'
            return
        if self.keep_time < 0:
            self.keep_time = self.ACTIONS_DUR[self.ctrl_signal['lat']]
            self.action_marking = self.ctrl_signal['lat']
            self.exec_lat_act()
        # if self.keep_time_index >= 0 and  self.keep_time_index < self.keep_time:
        #     self.LDWS()
        #     self.keep_time_index += 1
        #     return -1
        elif self.keep_time_index >= self.keep_time:
            # self.gl.scp.auto(self.gl.adv.name)
            self.keep_time = -1
            self.keep_time_index = -1
            self.action_marking = ''
            return -1
        self.keep_time_index += 1

 


    def vtd_exec_atomic(self):
        show_log = 0
        # ctrl     {"lat":"LANE_LEFT", "lon":FASTER}
        

        # while True:
        self.bMsgComplete = False
        self.gl.clear_fram_data()

            # while not bMsgComplete:
        while not self.bMsgComplete:
            # data = self.tcp_server.recv(204800)
            data = self.tcp_server.recv(409600)
            # data = self.tcp_server.recv(1024000)
            
            start_time  = time.time()
            
            ret  = len(data)
          #  print("********************************3*********************************",ret)

            if ret <=0:
                # print("recv error!!! ret < 0!!!")
                return 
            # # print("ret:",ret)
            # # print("bytesInBuffer + ret:",bytesInBuffer + ret)
            # # print("bufferSize:",bufferSize)

            pData = 0
            # if bytesInBuffer + ret > bufferSize:
                
            #     bufferSize = bytesInBuffer + ret
            # if ret!= msg[2] + msg[3]:
            #     ret = msg[2] + msg[3]
            # # print("ret:",ret)
        
            self.bytesInBuffer += ret
          #  print("bytesInBuffer:",self.bytesInBuffer)

            if  self.bytesInBuffer >=  SIZE_RDB_MSG_HDR_t:

                self.handle.update(struct.unpack('HHIIId',data[pData:pData + 24]) )


                if self.handle.magicNo != RDB_MAGIC_NO:
                    self.ytesInBuffer = 0

                while self.bytesInBuffer >= ( self.handle.headerSize + self.handle.dataSize):
                    # print("aaa",self.bytesInBuffer ,( self.handle.headerSize + self.handle.dataSize))
                    # # print("bytesInBuffer:",bytesInBuffer)
                    msgSize =  self.handle.headerSize + self.handle.dataSize
                    if show_log:
                       print("bytesInBuffer:",ret,self.bytesInBuffer,self.handle.headerSize,msgSize)

                    # # print("msgSize:",msgSize)
                    # isImage = False
                    # 48195
                    if self.bytesInBuffer == ( self.handle.headerSize + self.handle.dataSize):
                        self.gl.fram_data['simFrame'] = self.handle.frameNo
                        self.gl.fram_data['simTime'] = self.handle.simTime
                        # print("every fram:",self.gl.fram_data['simFrame'],self.gl.fram_data['simTime'])
                        self.parseRDBMessage(pData,data, self.handle)
                    pData += msgSize
                    self.bytesInBuffer -= msgSize
                    self.bMsgComplete = True
                    # update handle
                    if self.bytesInBuffer > self.handle.headerSize :
                        try:
                            self.handle.update(struct.unpack('HHIIId',data[pData:pData + 24]) )
                        except:
                            print("prare failed!!!")
                            continue

                    #     # inference time 10 ms
                # end_time = time.time()
                #  print('running times:',end_time - start_time)

                # 转换为全局坐标系
                if self.gl.adv is not None:
                    self.gl.adv.to_dict(self.gl.ego, True, True,True,name=1)
                index = 2
                for i in self.gl.objects:
                    i.to_dict(self.gl.ego, True,True,True,name=index)
                    index+=1

    def get_sqrt(self,x,y):
        return math.sqrt(math.pow(x,2) + math.pow(y,2))

    def run(self,action):
        self.vtd_exec_atomic(action)
    # return array 1* 36
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
    
    
    def get_dqn_state(self,left_lane = 1,right_lane = 1):
        traj_obs = []
        ita_obs = []
        if self.gl.adv is not None:
            left_lane  = 1 if abs(self.gl.adv.leftLaneId) < 30 else 0
            right_lane = 1 if abs(self.gl.adv.rightLaneId) < 30 else 0
            traj_obs = [self.model_manager.normalize_obs( self.gl.adv.vx,'vx'),\
                 self.model_manager.normalize_obs(self.gl.adv.vy,'vy'), \
                 self.gl.adv.pos_h, self.gl.adv.lane_offset,self.gl.adv.pos_h  + self.gl.adv.hdg, left_lane, right_lane]
            ita_obs +=  traj_obs
        if self.in_juction:
            ego_center = np.array(self.gl.ego.lane.map_vector)[:, :2].tolist()
            ego_lane_width = self.gl.ego.lane.map_vector[0][2] - 0.1
            bv_center = np.array(self.gl.adv.lane.map_vector)[:, :2].tolist()
            bv_lane_width = self.gl.adv.lane.map_vector[0][2] - 0.1
            # if np.linalg.norm(np.array(ego_center[0]) - np.array(bv_center[0])) <= 1:
            #     return 
        
            intersection, intersection_center = check_lane_intersection(ego_center, ego_lane_width, bv_center, bv_lane_width)
            if intersection and self.gl.adv.lane.lane_index != self.gl.ego.lane.lane_index:
                self.in_juction = True
                self.meeting_points = np.array(intersection_center[len(intersection_center) // 2])
                ego_lon, _ = self.gl.ego.lane.local_coordinates(self.meeting_points)
                ego_current_lon,_ = self.gl.ego.lane.local_coordinates(np.array([self.gl.ego.pos_x, self.gl.ego.pos_y]))
                bv_lon, _ = self.gl.adv.lane.local_coordinates(self.meeting_points)
                bv_current_lon , _= self.gl.adv.lane.local_coordinates(np.array([self.gl.adv.pos_x, self.gl.adv.pos_y]))
                ego_speed = self.gl.ego.get_sqrt(self.gl.ego.vx, self.gl.ego.vy) 
                adv_speed = self.gl.adv.get_sqrt(self.gl.adv.vx, self.gl.adv.vy)
                ego_t = (ego_lon - ego_current_lon) / ego_speed if ego_speed > 0.1 else 0.1
                bv_t = (bv_lon - bv_current_lon) / adv_speed if adv_speed > 0.1 else 0.1
                ita_obs.append(bv_t - ego_t )
                # ita_obs.append(-(ego_lon - ego_current_lon))
                ita_obs.append(bv_lon -  bv_current_lon )
            else:
                self.in_juction = False
        if self.gl.ego is not None:
            dx = self.gl.adv.pos_x -  self.gl.ego.pos_x
            dy = self.gl.adv.pos_y - self.gl.ego.pos_y
            dvx = self.gl.adv.vx - self.gl.ego.vx
            dvy = self.gl.adv.vy - self.gl.ego.vy
            traj_obs += [ self.model_manager.normalize_obs(dx,'x'),self.model_manager.normalize_obs( dy,'y') ,\
                self.model_manager.normalize_obs( dvx,'vx'),self.model_manager.normalize_obs( dvy,'vy'),\
                self.gl.ego.pos_h,  self.gl.ego.lane_offset, self.gl.ego.pos_h + self.gl.ego.hdg, 1, 1 ]
            ita_obs +=  [ self.model_manager.normalize_obs(dx,'x'),self.model_manager.normalize_obs( dy,'y') ,\
                self.model_manager.normalize_obs( dvx,'vx'),self.model_manager.normalize_obs( dvy,'vy'),\
                self.gl.ego.pos_h,  self.gl.ego.lane_offset, self.gl.ego.pos_h + self.gl.ego.hdg, 1, 1 ]
        
        for i in range(5):
            if i < len(self.gl.objects):
                traj_obs += [ self.model_manager.normalize_obs( self.gl.objects[i].pos_x ,'x'), self.model_manager.normalize_obs(self.gl.objects[i].pos_y,'y'),\
                    self.model_manager.normalize_obs(self.gl.objects[i].vx,'vx'), self.model_manager.normalize_obs(self.gl.objects[i].vy,'vy')  ]
            else:
                traj_obs += [0,0,0,0]

        traj_obs =  np.array(traj_obs).astype(self.space().dtype)
        ita_obs = np.array(ita_obs).astype(self.space().dtype)
        return torch.tensor(traj_obs).unsqueeze(0),torch.tensor(ita_obs).unsqueeze(0)
    
    def sample_action(self, state):
        
        with torch.no_grad():
            actions = self.value_net(state.to(self.type).to(self.device))
            actions_lat = self.value_net_lat(state.to(self.type).to(self.device))
            return actions, actions_lat
    def sample_actions(self,state,ita_state):
        # 使用训练好的模型进行动作选择
        with torch.no_grad():
            if self.in_juction:
                # ita_obs = np.random.rand(18).astype(self.space().dtype)
                # ita_obs = [0.33333333333333326, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.17994232725959, 8.644660822675124, -0.6007389418069842, -0.22916666666666663, 0.6333333333333333, 0.0, 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0]
                # # tensor([[ 1.8169e-02, -2.6485e-04, -1.4320e-02,  1.9289e-01,  4.2375e-07, 0.0000e+00,  0.0000e+00, -1.4369e+01,  9.4258e+00, -2.7060e-01,1.6031e-01,  1.9341e-02, -2.1530e-02,  1.6249e+00, -3.6091e-01, 1.5708e+00,  1.0000e+00,  1.0000e+00]])
                # ita_obs = [-0.3333,  0.0000,  3.1416,  0.0000,  3.1416,  0.0000,  0.0000, -0.2007, 18.0629,  0.4156, -0.2396, -0.3333,  0.3000, -1.5708,  0.0000, -1.5708, 0.0000,  0.0000]
                # ita_obs = np.array(ita_obs).astype(self.space().dtype)
                # ita_obs = torch.tensor(ita_obs).unsqueeze(0)
                # print('ita_state:',ita_state)
                # print('len ita_state:',len(ita_state[0]))
                lon_action = self.model_manager.model.value_net_ita( ita_state )
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
    def contrl(self):
        self.contrl_adv()
        if self.lib.get_msg_num()  ==  0 and self.gl.adv is not None:
            self.lib.addPkg(   self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, self.gl.adv.id, 1)
        print(">>>>>>>>>>>>>>>>>>>>ctrl>>>>>>>>>>>>>>>>>>>>:",self.ctrl_signal)


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
        print("adv:",self.gl.adv.name)
        if self.gl.front_vec_to_compete is not None:
            print("front_vec_to_compete:",self.gl.front_vec_to_compete.name)
        # print("left_neib_front_vec_to_compete:",self.gl.left_neib_front_vec_to_compete.name)
        # print("right_neib_front_vec_to_compete:",self.gl.right_neib_front_vec_to_compete.name)
        if self.gl.bake_vec_to_compete is not None:
            print("bake_vec_to_compete:",self.gl.bake_vec_to_compete.name)
        # print("left_neib_bake_vec_to_compete:",self.gl.left_neib_bake_vec_to_compete.name)
        # print("right_neib_bake_vec_to_compete:",self.gl.right_neib_bake_vec_to_compete.name)
        if self.gl.front_vec_to_compete is not None:
            # self.gl.front_vec_to_compete.show()
            if self.gl.front_vec_to_compete.vx < 0:
                ttc =self.get_speed(self.gl.front_vec_to_compete.pos_x - 5, self.gl.front_vec_to_compete.pos_y)  /  self.get_speed(self.gl.front_vec_to_compete.vx, self.gl.front_vec_to_compete.vy)
                if ttc < 4:
                    print("collision warning!!!  ttc:",ttc)
                    return True
        # for i in other_vecs:
        #     i.show()
        return False

    def stop(self):
        self.lib.clear()
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -4, 0, self.gl.adv.id  , 1)   

    def sendctrlmsg(self):
        self.lib.sendTrigger(self.sClient, self.gl.fram_data['simTime'], self.gl.fram_data['simFrame'],0  )
        self.lib.clear()
    
    def autopilot(self,actor_id):
        self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, actor_id  , 0)   
        # 

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

    def vtd_func(self):

        ego_time,ego_s, ego_v = self.get_time(self.gl.ego)

        for i in self.gl.objects:

            time,s,v = self.get_time(i)
            time +=0
            
            if time < 0:
                continue

            _ , in_juction = self.chek_in_juction( i.roadId, 0,i.lane_id)
        #    if self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == 'STOP':
        #         print("decrease acc!!!",i.id)
        #         if  self.get_sqrt( i.vx,  i.vy) < 1:
        #             self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -0.5, 0, i.id, 1)
        #         else :
        #             self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, i.id, 1)
            if self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == 'STOP':
                print("decrease acc!!!",i.id)
                if  self.get_sqrt( i.vx,  i.vy) > 0:
                    self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -3, 0, i.id, 1)
                else :
                    self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 0, 0, i.id, 1)
            # if self.in_juction is False  and  in_juction is False and self.gl.ego.light_state == '':
            elif  time > ego_time:
                self.lib.addPkg(  self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], 1.5, 0, i.id, 1)
            elif time <= ego_time:
                print("decrease acc!!!")
                self.lib.addPkg( self.gl.fram_data["simTime"], self.gl.fram_data["simFrame"], -5, 0, i.id, 1)

    def run_loop(self,ctrl):  
        self.ctrl_signal =  ctrl
        for i in range(5):
            self.vtd_exec_atomic()
        while 1 :
            self.vtd_exec_atomic()
            # 选定对抗车
            if self.gl.adv is None and len(self.gl.objects) > 0:
                self.vtd_func()

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
                    state,ita_state = self.get_dqn_state()
                    # 获取模型输出指令
                    self.ctrl_signal = self.sample_actions(state,ita_state)
                    # 碰撞检测，如果有碰撞危险返回True, 此处会将所有目标转换为对抗车辆坐标系
                    if self.collision_warning():
                        # 刹车
                        self.stop()
                    else:
                        # 控制指令
                        self.contrl()
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

class PREPARE():
    def __init__(self,scenario_path = None):

        if scenario_path  is None :
            data_path = '/data/'
            for i in os.listdir(data_path):
                if i[-3:] == 'xml':
                    # print(i)
                    scenario_path = os.path.join(data_path, i)
        self.scenario_path = scenario_path
        self.confrontation_position = [-1,-1]
        self.get_confrontation()

    def get_confrontation(self):
        tree = ET.parse(self.scenario_path)
        root = tree.getroot()
        for node in root.iter("Scenario"):
            for node1 in node.iter():
                if node1.tag == 'Action':
                    if node1.attrib['Name'] == 'confrontation1':
                        # print("node tag:",node1.tag)
                        # print("attrib:",node1.attrib)
                        for node2 in node1.iter():
                            if node2.tag == 'PosAbsolute':
                                # print("node tag:",node2.tag)
                                # print("attrib:",node2.attrib)
                                print("confrontation position:")
                                print("x:",float( node2.attrib['X']))
                                print("y:", float(node2.attrib['Y']))
                                self.confrontation_position[0] = float( node2.attrib['X'] )
                                self.confrontation_position[1] = float( node2.attrib['Y'] ) 

class Model():
    def __init__(self,args,output_path,model = 'dqn') -> None:
        self.args = args
        self.device=torch.device('cpu')
        self.output_path = output_path
        self.model = model
        self.load_model_id = args.load_model_id
        self.load_model_id_lat = args.load_model_id_lat
        self.load_model_id_ita = args.load_model_id_ita
        self.train = False
        self.lr = 0.001
        
        self.init_network()
        self.load_model(self.load_model_id,self.load_model_id_lat,self.load_model_id_ita)
        
    def init_network(self):
        """
        init_network方法用于初始化网络。根据args.update_mode的不同取值，创建了不同的网络模型（Net或DuelingDQN），
        并将网络模型和优化器存储在self.value_net、self.target_net和self.optimizer中。
        """
        if self.model == 'dqn':
            self.value_net = Net(self.args.state_dim, self.args.action_dim)
            self.target_net = Net(self.args.state_dim, self.args.action_dim)
            to_device(self.device, self.value_net)
            self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)


            self.value_net_lat = Net(self.args.state_dim_lat, self.args.action_dim_lat)
            self.target_net_lat = Net(self.args.state_dim_lat, self.args.action_dim_lat)
            to_device(self.device, self.value_net_lat)
            self.optimizer_lat = torch.optim.Adam(self.value_net_lat.parameters(), lr=self.lr)

            self.value_net_ita = Net(self.args.state_dim_ita, self.args.action_dim_ita)
            self.target_net_ita = Net(self.args.state_dim_ita, self.args.action_dim_ita)
            to_device(self.device, self.value_net_ita)
            self.optimizer_ita = torch.optim.Adam(self.value_net_ita.parameters(), lr=self.lr)

        elif self.model == 'ddqn':
            self.value_net = DuelingDQN(self.args.state_dim, self.args.action_dim)
            self.target_net = DuelingDQN(self.args.state_dim, self.args.action_dim)
            to_device(self.device, self.value_net)
            self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)

    def load_model(self, epoch, epoch_lat,epoch_ita):
        # 加载模型权重
        if self.train:
            model_path = os.path.join(self.output_path, f'latest_model_ckpt.pth')
        else:
            model_path = os.path.join(self.output_path, f'{epoch}_model_ckpt.pth')
            model_path_lat = os.path.join(self.output_path, f'{epoch_lat}_model_ckpt.pth')
            model_path_ita = os.path.join(self.output_path, f'{epoch_ita}_model_ckpt.pth' )
        Utils.print_banner(f'Load model from {model_path}.')
        Utils.print_banner(f'Load model from {model_path_lat}.')
        assert os.path.exists(model_path), print(model_path)
        assert os.path.exists(model_path), print(model_path_lat)
        ckpt = torch.load(model_path, map_location=self.device)
        ckpt_lat = torch.load(model_path_lat, map_location=self.device)
        ckpt_ita = torch.load(model_path_ita, map_location=self.device)
        if self.model == 'dqn' or 'ddqn':
            self.value_net.load_state_dict(ckpt['value_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])

            self.value_net_lat.load_state_dict(ckpt_lat['value_net'])
            self.target_net_lat.load_state_dict(ckpt_lat['target_net'])
            self.optimizer_lat.load_state_dict(ckpt_lat['optimizer'])

            self.value_net_ita.load_state_dict(ckpt_ita['value_net'])
            self.target_net_ita.load_state_dict(ckpt_ita['target_net'])
            self.optimizer_ita.load_state_dict(ckpt_ita['optimizer'])

class ModelManager():
    def __init__(self,args, output_path, model = 'dqn'): 
        self.model= Model(args,output_path,model= model)
        self.features_range = {'x': [-60, 60], 'y': [-60, 60], 'vx': [-30, 30], 'vy': [-30, 30]}
    def observe():
        pass
    def normalize_obs(self,x, x_name = 'x'):
        return  utils.lmap(x, [self.features_range[x_name][0], self.features_range[x_name][1]], [-1, 1])
    # state
    def get_action(state):
        pass



def make_args():
    ca = True
    xodr_path = None
    if ca:
        data_path = '/data/'
        for i in os.listdir(data_path):
            if i[-4:] == 'xodr':
                # print(i)
                xodr_path = os.path.join(data_path, i)

    config = ['ramp', 'exp_6', 
            {'AtomicScene': 'all', 'map_file_path': xodr_path,
            'LON_ACTIONS':{
                0: 'IDLE', 
                1: 'FASTER', 
                2: 'SLOWER'} ,
            "LAT_ACTIONS": {
                0: "LANE_LEFT",
                1: "IDLE",
                2: "LANE_RIGHT",
                3: "LEFT_1",
                4: "LEFT_2",
                5: "RIGHT_1",
                6: "RIGHT_2"}
            },
            # '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/ca/changan_adv_model_1_10/data/model/model/roundabout'
            '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/ca/changan_adv_model_1_10/data/model/model/ramp/lanechange'
            # '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/ca/changan_adv_model_1_10/data/model/DQNAgent_2_2/intersection/'
            # '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/ca/changan_adv_model_1_10/data/model/dqn_model_inference/'
            ]
    # '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/ca/changan_adv_model_1_10/data/model/dqn_model_inference/intersection'
    args = ArgumentParser()
    args.add_argument('--model', default="adv model", type=str, help=""" It's one of three:ppo, trpo, a2c.""")
    args.add_argument('--scenario', default=config[0], type=str, help="""ramp, t_intersection, intersection""")

    args.add_argument('--exp-name', default=config[1], type=str)
    args.add_argument('--action-type', default='discrete', type=str)
    args.add_argument('--update-mode', default='dqn', type=str)
    args.add_argument('--seed', default=0, type=int)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # zadao lon :454    lat 370 
    args.add_argument('--state_dim', default=36, type=int)  
    args.add_argument('--action_dim', default=3, type=int)
    args.add_argument('--state_dim_lat', default=36, type=int)
    args.add_argument('--action_dim_lat', default=7, type=int)
    args.add_argument('--state_dim_ita', default=18, type=int)
    args.add_argument('--action_dim_ita', default=3, type=int)
    args.add_argument('--env_config', default=config[2], type=dict) # lat_mode lon_mode
    args.add_argument('--output_path', default=config[3]) # lat_mode lon_mode
    args.add_argument('--vehicles_count', default=6) # lat_mode lon_mode
    model = 4
    if model == 1:
        # intersection
        args.add_argument('--load-model-id', default=490, type=int) # 454  374
        args.add_argument('--load-model-id-lat', default=469, type=int) # exp4 400 407 exp5 417 455  370
        args.add_argument('--load-model-id-ita', default=306, type=int) # exp4 400 407 exp5 417 455  370


    if model == 2:
        # # ramp 454  374
        args.add_argument('--load-model-id', default=454, type=int) # 454  374
        args.add_argument('--load-model-id-lat', default=370, type=int) # exp4 400 407 exp5 417 455  370
        args.add_argument('--load-model-id-ita', default=434, type=int) # exp4 400 407 exp5 417 455  370
    if model == 3:
        # roundabout
        args.add_argument('--load-model-id', default=363, type=int) # 454  374
        args.add_argument('--load-model-id-lat', default=485, type=int) # exp4 400 407 exp5 417 455  370
        args.add_argument('--load-model-id-ita', default=434, type=int) # exp4 400 407 exp5 417 455  370
    if model == 4:
        # lanechange
        args.add_argument('--load-model-id', default=482, type=int) # 454  374
        args.add_argument('--load-model-id-lat', default=376, type=int) # exp4 400 407 exp5 417 455  370
        args.add_argument('--load-model-id-ita', default=434, type=int) # exp4 400 407 exp5 417 455  370   
    return args


if __name__ == '__main__':

    ctrl = {'lon':'IDLE', 'lat':'RIGHT_2'}    
    args = make_args().parse_args()
    mm = VTD_Manager(open_vtd=True,test=True,args=args)
    mm.gl.scp.start_scenario(scenario_file=mm.scenario.scenario_path)
    mm.run_loop(ctrl)
    # pre = PREPARE()
    # print("confrontation_position:",pre.confrontation_position)
    # adv_speed = 0.6343014050077045
    # var = -1
    # speed_index = mm.get_speed_index(adv_speed)
    # target_speed = mm.DEFAULT_TARGET_SPEEDS[ speed_index +var ]g
    # print(mm.DEFAULT_TARGET_SPEEDS)
    # print("speed_index/target_speed",speed_index, target_speed)
