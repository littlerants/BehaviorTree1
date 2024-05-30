import socket
import struct

# SCP发送指令类
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

    # 开始某个仿真任务，需要配置xml

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
    # 停止仿真
    def stop(self):
        data = "<SimCtrl><Stop /></SimCtrl>"
      #  print("stop.....")
        handle = self.get_handle(data)
        self.send(handle)
    # 获取scp报文句柄
    def get_handle(self,data):

        return  struct.pack('HH64s64sI{}s'.format(len(data)),40108,0x0001,"ExampleConsole".ljust(64,'\0').encode('utf-8'),"any".ljust(64,'\0').encode('utf-8') ,len( data ),data.encode('utf-8'))
    # 发送scp报文句柄
    def send(self,handle):
        self.tcp_server_inf.send(handle)

    # 车辆右变道指令
    def turn_right(self,actor,num = -1):
        data = "<Traffic><ActionLaneChange direction='{}' force='true' delayTime='0.0' approveByDriver='false' activateOnExit='false' driverApproveTime='0' actor='{}' time='3.0'/></Traffic>"
        data = data.format(num,actor)
        handle = self.get_handle(data)
        self.send(handle)
    # 这两左变道指令
    def turn_left(self,actor,num = 1):
        data = "<Traffic><ActionLaneChange direction='{}' force='true' delayTime='0.0' approveByDriver='false' activateOnExit='false' driverApproveTime='0' actor='{}' time='3.0'/></Traffic>"
        data = data.format(num,actor)
        handle = self.get_handle(data)        
        self.send(handle)

    # 车辆加减速指令
    def dacc(self,actor, target_speed = 20,type = None):
        # set ped's speed if type == 5
        # logging.debug("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
        # 支持行人加减速
        if type is not None and type == 5:
            # m/s -> km/h
            # target_speed *= 3.6
            if target_speed > 5:
                target_speed == 5
            if target_speed < -5:
                target_speed == -5
            data = "<Traffic><ActionMotion speed='{}' actor='{}'  rate='2' force='true' delayTime='0'/></Traffic>"
        else:
            data  = "<Traffic><ActionSpeedChange rate='3' target='{}' force='true' delayTime='0.0' activateOnExit='false' pivot='' actor='{}'/></Traffic>"
        data = data.format(target_speed, actor)
        handle = self.get_handle(data)        
        self.send(handle)
    # 自驾模式，由车辆由VTD default driver控制 
    def auto(self,actor):
        self.off_light(actor)

        data = "<Traffic> <ActionAutonomous enable='true' force='true' delayTime='0.0' activateOnExit='false' actor='{}'/> </Traffic>"
        
        data = data.format(actor)
        # logging.debug("data:{}".format(data))
        handle = self.get_handle(data)        
        self.send(handle)
    # 灯光控制
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
    # 位置瞬移
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
    # 车道偏移控制
    def Laneoffset(self,name,offset):
        data  =  "<Player name='{}'><LaneOffset absolute='{}' time='0' s='0'/></Player>".format(name,offset)
        handle = self.get_handle(data)        
        self.send(handle)
    # 车道偏移回退 or  车道居中
    def overLaneoffset(self,name):
        self.Laneoffset(name,0)
