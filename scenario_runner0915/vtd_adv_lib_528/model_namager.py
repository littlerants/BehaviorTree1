
import os
from Utils.torch import to_device
import torch
import Utils
from gym_sumo import utils

from models.dqn_net import Net, DuelingDQN

# 对抗模型初始化与加载类
class Model():
    def __init__(self,args,model_path,model = 'dqn') -> None:
        self.args = args
        self.device = torch.device('cpu')
        self.model_path = model_path
        self.model = model
        self.train = False
        self.lr = 0.001
        self.model_list = {}
        self.init_network()
        # self.load_model(self.load_model_id,self.load_wall_model_id,self.load_lon_model_id)

    def init_network(self):
        """
        init_network方法用于初始化网络。根据args.update_mode的不同取值，创建了不同的网络模型（Net或DuelingDQN），
        并将网络模型和优化器存储在self.value_net、self.target_net和self.optimizer中。
        """
        # if self.model == 'dqn':
        #     self.value_net = Net(self.args.state_dim_1, self.args.actions_dim_1)
        #     self.wall_value_net = Net(self.args.wall_state_dim, self.args.wall_actions_dim)
        #     self.lon_value_net = Net(self.args.wall_state_dim, self.args.wall_actions_dim)
        #
        #     to_device(self.device, self.value_net)
        #     to_device(self.device, self.wall_value_net)
        #     to_device(self.device, self.lon_value_net)
        # elif self.model == 'ddqn':
        #     self.value_net = DuelingDQN(self.args.state_dim_1, self.args.actions_dim_1)
        #     self.wall_value_net = DuelingDQN(self.args.wall_state_dim, self.args.wall_actions_dim)
        #
        #     to_device(self.device, self.value_net)
        #     to_device(self.device, self.wall_value_net)
        for key, val in self.args.model_config_list.items():
            if val.depart:
                pass
            else:
                self.model_list[key] = Net( val.state_dim,len(val.actions),self.args.model_config_list[key].hidden_size )
            if val.file_name:
                ab_model_path =  os.path.join(self.model_path, f'{val.file_name}.pth')    
                print("Load model ",key,"  from  ",ab_model_path)
                print("----------------------------------------------------------")
                ckpt = torch.load(ab_model_path, map_location=self.device)
                self.model_list[key].load_state_dict(ckpt['value_net'])

    # def load_model(self, epoch,wall_epoch,lon_epoch):
    #     # 加载模型权重
    #     if epoch:
    #         model_path = os.path.join(self.model_path, f'{epoch}.pth')
    #
    #         Utils.print_banner(f'Load lon model from {model_path}.')
    #         assert os.path.exists(model_path), print(model_path)
    #         ckpt = torch.load(model_path, map_location=self.device)
    #         if self.model == 'dqn' or 'ddqn':
    #             self.value_net.load_state_dict(ckpt['value_net'])
    #     if wall_epoch:
    #         wall_model_path = os.path.join(self.model_path, f'{wall_epoch}.pth')
    #         Utils.print_banner(f'Load lon model from {wall_model_path}.')
    #         assert os.path.exists(wall_model_path), print(wall_model_path)
    #         wall_ckpt = torch.load(wall_model_path, map_location=self.device)
    #         if self.model == 'dqn' or 'ddqn':
    #             self.wall_value_net.load_state_dict(wall_ckpt['value_net'])
    #     if lon_epoch:
    #         lon_model_path = os.path.join(self.model_path, f'{lon_epoch}.pth')
    #         Utils.print_banner(f'Load lon model from {lon_model_path}.')
    #         assert os.path.exists(lon_model_path), print(lon_model_path)
    #         lon_ckpt = torch.load(lon_model_path, map_location=self.device)
    #         if self.model == 'dqn' or 'ddqn':
    #             self.lon_value_net.load_state_dict(lon_ckpt['value_net'])


# 对抗模型管理类
class ModelManager():
    def __init__(self,args, model_path, model = 'dqn'):
        self.model_config_list = args.model_config_list
        self.model= Model(args, model_path,model= model)
        self.features_range = {'x': [-60, 60], 'y': [-60, 60], 'vx': [-30, 30], 'vy': [-30, 30]}
    def observe():
        pass
    def normalize_obs(self,x, x_name = 'x'):
        return  utils.lmap(x, [self.features_range[x_name][0], self.features_range[x_name][1]], [-1, 1])
    # state
    def get_action(state):
        pass
