# from argparse import ArgumentParser
# from vtd_adv_lib_529.vtd_manager_513 import ADV_Manager
import os
# import sys
# # sys.path.insert(0,os.path.join(os.getenv('ADVPATH'),'gym_sumo'))
# # sys.path.insert(0,  os.getenv('ADVPATH') )
# os.environ['ADVPATH'] = os.getcwd().replace('\\','/')
# sys.path.insert(0,os.environ['ADVPATH'])
# print("sys path",sys.path)
# os.environ['ADVPATH'] = "/home/vtd/VTD.2.2/simulate_VTD_ADV_traffic/"
# os.environ['ADVPATH'] = "/home/zjx/work/BehaviorTree1/scenario_runner0915/"
# # os.environ['ADVPATH'] = "/home/zjx/work/trafficflow/cloud/simulation-executor/scenario_runner/srunner/scenarios/traffic_flow/adv_lib/"
# print("sys path",sys.path)

class MODEL_CONFIG:
    def __init__(self,config):
        self.config = config
        self.name = config['name']
        self.file_name = config['file_name']
        self.depart = config['depart']
        self.hidden_size = config['hidden_size']
        if self.depart:
            self.state_dim_lon = config['state_dim_lon']
            self.state_dim_lat = config['state_dim_lat']
            self.actions_lon = config['actions_lon']
            self.actions_lat = config['actions_lat']
        else:
            self.state_dim = config['state_dim']
            self.actions = config['actions']



class CONFIG:
    def __init__(self):
        self.xodr_path = None
        self.scenario_path = None
        self.model_path = None
        self.init()
        self.output_path = None

        self.model_config_list = {}

        model_config_no_wall = {"name":"model_no_wall", "file_name":"newest_no_wall", "depart" : False,"hidden_size":(128,256), "state_dim" : 11, "actions":{0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER', 5: "LEFT_2",6: "RIGHT_2"} }
        # 1328_model_ckpt   577_model_ckpt    831_model_ckpt  867_model_ckpt
        model_no_wall = MODEL_CONFIG(model_config_no_wall)
        self.model_config_list[model_no_wall.name] = model_no_wall

        model_config_wall = {"name":"model_wall", "file_name":"has_wall_exp_15_id_847", "depart" : False,"hidden_size":(128,256), "state_dim" : 16, "actions":{0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER', 5: "LEFT_2",6: "RIGHT_2"} }
        # 999_model_ckpt
        model_wall = MODEL_CONFIG(model_config_wall)
        self.model_config_list[model_wall.name] = model_wall

        model_config_lon = {"name":"model_lon","file_name":"model_lon", "depart":False,"hidden_size":(256,256), "state_dim" : 11, "actions":{0: 'IDLE', 1: 'FASTER', 2: 'SLOWER'} }
        model_lon = MODEL_CONFIG(model_config_lon)
        self.model_config_list[model_lon.name] = model_lon

        model_config_lon = {"name":"model_dynamic_wall","file_name":"1447_model_ckpt", "depart":False,"hidden_size":(128,256), "state_dim" : 16, "actions":{0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER', 5: "LEFT_2",6: "RIGHT_2"}   }
        model_lon = MODEL_CONFIG(model_config_lon)
        self.model_config_list[model_lon.name] = model_lon
    def init(self):
        ca = False
        xodr_path = None
        if ca:
            data_path = '/data/'
            for i in os.listdir(data_path):
                if i[-4:] == 'xodr':
                    # print(i)
                    self.xodr_path = os.path.join(data_path, i)
                if i[-3:] == 'xml':
                    # print(i)
                    self.scenario_path = os.path.join(data_path, i)
        self.model_path = os.path.join(os.getenv('ADVPATH'), 'adv_model')

#
# if __name__ == '__main__':
#
#     ctrl = {'lon':'IDLE', 'lat':'RIGHT_2'}
#     args = CONFIG()
#     mm = ADV_Manager(open_vtd=True,test=True,args=args)
#     # mm.gl.scp.start_scenario(scenario_file=mm.scenario.scenario_path)
#     mm.run_loop(ctrl)
#     # pre = PREPARE()
#     # print("confrontation_position:",pre.confrontation_position)
#     # adv_speed = 0.6343014050g077045
#     # var = -1g
#     # speed_index = mm.get_speed_index(adv_speed)
#     # target_speed = mm.DEFAULT_TARGET_SPEEDS[ speed_index +var ]g
#     # print(mm.DEFAULT_TARGET_SPEEDS)
#     # print("speed_index/target_speed",speed_index, target_speed)





