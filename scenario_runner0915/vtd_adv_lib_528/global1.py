


from vtd_adv_lib.scp import SCP

# 全局类，当前类主要负责仿真环境中目标障碍物等的数据存储，融合和管理
class GLOBAL():
    def __init__(self):
        self.scp = SCP()
        # vecs_set = []
        self.objects_set = []
        self.objects = []
        self.result = {}
        self.ego = None
        self.adv = None
        self.pre_adv_lane_id = None
        self.open_lat = False
        # 对抗车辆前后，左侧前后，右侧前后车辆
        self.left_neib_front_vec_to_compete = None
        self.left_neib_bake_vec_to_compete = None
        self.right_neib_front_vec_to_compete = None
        self.right_neib_bake_vec_to_compete = None
        self.front_vec_to_compete = None
        self.bake_vec_to_compete = None
        self.compete_time = 0
        self.compete_time_range = 300
        self.last_compete_name= ''
        self.last_compete_name_id = -1
        self.show_log = True
        self.close_light = True
        self.fram_data = {"simFrame":0, "simTime":0,
            "Objects":[],"RoadPos":[],"LaneInfo":{},
            "RoadState":[],"ROADMARK":[],"TrafficSign":[],"TRAFFIC_LIGHT":[]}
        self.ego_state = {}
        # 目标加速度
        self.target_acc = 0
        self.slow_dur = 0
        self.adv_pre_speed = 0
        self.adv_hdg_num = 0
        self.dangerous = 1
        self.lon_offset = 3
        self.adv_flag = True

    # 数据融合

    def fusion_rdb_data(self):
        for i in  self.fram_data["Objects"]:
            # print("name/id",i['name'],i['id'])
            if len(self.fram_data['RoadPos']): 
                for j in self.fram_data['RoadPos']:
                    if i['id'] == j['id']: 
                        i.update(j)
            if len(self.fram_data['RoadState']):
                for j2 in self.fram_data['RoadState']:
                    if i['id'] == j2['id']:
                        i.update(j2)
    # 数据清除
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
        self.target_acc = 0
        # self.vecs_set = []