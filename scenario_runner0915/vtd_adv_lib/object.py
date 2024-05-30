import math
from collections import deque

# 目标类
class OBJECT():
    def __init__(self,simTime = -1,simFrame = -1, name='',id=100,lane_id=0,pos_x=0,off_x=1.39,pos_y=0,pos_h=0,hdg=0,vx=0,vy=0,
                v_h=0,w=0,l=0,lane_offset=0,inertial_heading=0,lane_w=0,obj_type = '1',acc_x  = 0, acc_y = 0,
                leftLaneId = -1,rightLaneId = -3,  distToJunc=10000000000.0,light_state='GO',roadId=None,adv_vec=False,lane = None,wp=None):
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
        self.pos_trajx = deque(maxlen=20)
        self.pos_trajy = deque(maxlen=20)
        # 0 front
        # 1 back
        # 2 left
        # 3 right
        self.direction_to_ego = -1

        self.wp = wp


    # rotate      -1为顺时针方向
                # 1 为逆时针方向
    #   车辆坐标系转换为全局坐标系(现阶段以ego车为base)，方便强化学习模型计算
    def trans_cood1(self, base_obj, position_flag = False,velocity_flag = False,acc_flag = False ,rotate = -1):

        theta =rotate * base_obj.pos_h
        theta = (theta / 180) * 3.1415926
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
            # self.pos_h += base_obj.pos_h

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
    # 转换为车辆（对康车）坐标系，用以方便碰撞检测等
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

    def global_trans_local(self, base_obj, position_flag=False, velocity_flag=False, acc_flag=False, rotate=-1):

        theta = rotate * base_obj.pos_h
        theta = (theta/180)*3.1415926
        # if position_flag:
        #     pos_x = self.pos_x - base_obj.pos_x
        #     pos_y = self.pos_y - base_obj.pos_y
        #     self.pos_x = pos_x * math.cos(theta) + pos_y * math.sin(theta)
        #     self.pos_y = pos_y * math.cos(theta) - pos_x * math.sin(theta)
        #     # if heading:
        #     deta_heading = self.pos_h - base_obj.pos_h
        #     if deta_heading >= 0:
        #         self.pos_h = deta_heading if abs(deta_heading) < 3.1415926 else deta_heading - 6.2831852
        #     else:
        #         self.pos_h = deta_heading if abs(deta_heading) < 3.1415926 else deta_heading + 6.2831852

        if velocity_flag:
            vx = self.vx - base_obj.vx
            vy = self.vy - base_obj.vy
            self.vx = vx * math.cos(theta) + vy * math.sin(theta)
            self.vy = vy * math.cos(theta) - vx * math.sin(theta)
        if acc_flag:
            acc_x = self.acc_x - base_obj.acc_x
            acc_y = self.acc_y - base_obj.acc_y
            # print("org acc:",self.acc_x,self.acc_y)
            self.acc_x = acc_x * math.cos(theta) + acc_y * math.sin(theta)
            self.acc_y = acc_y * math.cos(theta) - acc_x * math.sin(theta)


                    # print("trans acc:",self.acc_x,self.acc_y)
    # 更新所有参数
    def update(self,simTime=-1, simFrame=-1,lane_id=None,pos_x=None,off_x = None, pos_y=None,pos_h=None,hdg=None,vx=None,vy=None,
                v_h=None,w=None,l=None,lane_offset=None,inertial_heading=None,lane_w=None,obj_type = None,acc_x  = None, acc_y = None,
                leftLaneId = None,rightLaneId = None,  distToJunc=None,light_state=None,roadId=None,lane = None,wp = None):
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
        if wp != None:
            self.wp = wp
    def update1(self,obj):
        self.simTime = obj.simTime
        self.simFrame = obj.simFrame
        self.name = obj.name
        self.id = obj.id
        self.lane_id =obj.lane_id
        self.pos_x = obj.pos_x
        self.off_x = obj.off_x
        self.pos_y = obj.pos_y
        self.pos_h = obj.pos_h
        self.vx = obj.vx
        self.vy = obj.vy
        self.v_h = obj.v_h
        self.hdg = obj.hdg
        self.acc_x = obj.acc_x
        self.acc_y = obj.acc_y
        self.w = obj.w
        self.l = obj.l
        self.lane_offset = obj.lane_offset
        self.inertial_heading = obj.inertial_heading
        self.lane_w = obj.lane_w
        self.distToJunc = obj.distToJunc
        self.obj_type = obj.obj_type
        self.light_state = obj.light_state
        self.leftLaneId = obj.leftLaneId
        self.rightLaneId = obj.rightLaneId
        self.roadId = obj.roadId
        # self.predisToconfrontation_position = 99999999
        self.new_object = False
        self.adv_vec = obj.adv_vec
        self.lane = obj.lane
        self.wp = obj.wp



    def show(self):
       print("simTime:",self.simTime,"simFrame:",self.simFrame,"name:",self.name,"id:",self.id,"x/y:",self.pos_x, self.pos_y, "vx/vy:",self.vx,self.vy, "pos_h:",self.pos_h ,"hdg:",self.hdg,"acc_x/y:",self.acc_x,self.acc_y,"roadId:",self.roadId, "lane_id:",self.lane_id )
    # 车辆坐标系转换为全局坐标系
    def to_dict(self,ego = None, position_flag = False, velocity_flag = False,acc_flag = False, name = None):
        if ego is not  None: 
            self.trans_cood1(ego,position_flag,velocity_flag,acc_flag )
        
        if name is None:
            return {self.name:{"simTime":self.simTime,"simFrame":self.simFrame,"name":self.name,"id":self.id, "pos_x":self.pos_x,"pos_y":self.pos_y, "pos_h":self.pos_h, "vx":self.vx, "vy":self.vy}}
        else:
            return {name:{"simTime":self.simTime,"simFrame":self.simFrame,"name":self.name, "id":self.id,"pos_x":self.pos_x,"pos_y":self.pos_y, "pos_h":self.pos_h, "vx":self.vx, "vy":self.vy}}
    # 获取平方根
    def get_sqrt(self,x,y):
        return math.sqrt(x**2 + y**2)