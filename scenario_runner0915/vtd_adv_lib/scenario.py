
import os
import xml.etree.ElementTree as ET

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
    # 从xodr中获取相遇点坐标
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
                                print("x/y:","( ",float( node2.attrib['X']),",",float(node2.attrib['Y']), ")")
                                self.confrontation_position[0] = float( node2.attrib['X'] )
                                self.confrontation_position[1] = float( node2.attrib['Y'] ) 

# 场景管理类，
class SCENARIO():
    def __init__(self, data_path  =  None):
        # data_path = None
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
            flag = 3
            # t intersection
            if flag == 1:
                xodr_path     = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/dingzilukou/dingzilukou.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/dingzilukou/dingzilukou.xml'
            # zadao
            if flag == 2:
                xodr_path     = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/1027_01.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/1027_01.xml'
                xodr_path     = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/0228.xodr'
                
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/zadao/0228.xml'
            # intersection
            if flag == 3:
               
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/intersection/stopandgo+.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/intersection/stopandgo+.xml'
                # xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/test/stopandgo+.xodr'
                # scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Develop/test/stopandgo+.xml'
                # '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/intersection'
            # roundabout
            if flag == 4:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/SampleProject/Scenarios/zjx/vtd_scenarios/3round/round1.xodr' 
                scenario_path =  '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/SampleProject/Scenarios/zjx/vtd_scenarios/3round/round1.xml'
                
                # xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/roundabout/roundabout/Roundabout_with_5_Three_Way_exits-0001.xodr'
                # scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/roundabout/roundabout/Roundabout_with_5_Three_Way_exits-0001.xml'
            # lanechange
            if flag == 5:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/3_lane_to_2_Lane-0001.xodr'
                scenario_path ='/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/3_lane_change_2_lane.xml'
            if flag == 6:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/2_lane_to_1_Lane-0001.xodr'
                scenario_path ='/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/lanechange/2_lane_change_to_1_lane.xml'     
            if flag == 7:
                # xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/3round/Roundabout_with_3_Three_Way_exits-0001.xodr'
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/3round/Roundabout_with_3_Two_Way_exits-0001.xodr'
                scenario_path ='/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/3round/round1.xml'      
            if flag == 8:
                xodr_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/peds/stopandgo+.xodr'
                scenario_path = '/home/sda/upload/VTD.2021/VTD.2021.3/Data/Projects/Current/Scenarios/zjx/vtd_scenarios/peds/stopandgo+.xml'
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