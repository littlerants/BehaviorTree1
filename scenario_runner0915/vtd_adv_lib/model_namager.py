
import os

import torch


from vtd_adv_lib.models.dqn_net import Net, DuelingDQN

from vtd_adv_lib.Utils.torch import to_device
import vtd_adv_lib.Utils
from vtd_adv_lib.gym_sumo.gym_sumo import utils
# 对抗模型初始化与加载类
class Model():
    def __init__(self,args,output_path,model = 'dqn') -> None:
        self.args = args
        self.device=torch.device('cpu')
        self.output_path = output_path
        self.model = model
        self.load_model_id = args.load_model_id
        self.load_model_id_lat = args.load_model_id_lat
        self.load_model_id_ita = args.load_model_id_ita
        self.load_model_id_ped = args.load_model_id_ped
        self.train = False
        self.lr = 0.001
        
        self.init_network()
        self.load_model(self.load_model_id,self.load_model_id_lat,self.load_model_id_ita,self.load_model_id_ped)
        
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


            self.value_net_ped = Net(self.args.state_dim_ped, self.args.action_dim_ped)
            self.target_net_ped = Net(self.args.state_dim_ped, self.args.action_dim_ped)
            to_device(self.device, self.value_net_ped)
            self.optimizer_ped = torch.optim.Adam(self.value_net_ped.parameters(), lr=self.lr)

        elif self.model == 'ddqn':
            self.value_net = DuelingDQN(self.args.state_dim, self.args.action_dim)
            self.target_net = DuelingDQN(self.args.state_dim, self.args.action_dim)
            to_device(self.device, self.value_net)
            self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)

    def load_model(self, epoch, epoch_lat,epoch_ita , epoch_ped):
        # 加载模型权重
        if self.train:
            model_path = os.path.join(self.output_path, f'latest_model_ckpt.pth')
        else:
            model_path = os.path.join(self.output_path, f'{epoch}.pth')
            model_path_lat = os.path.join(self.output_path, f'{epoch_lat}.pth')
            model_path_ita = os.path.join(self.output_path, f'{epoch_ita}.pth' )
            model_path_ped = os.path.join(self.output_path, f'{epoch_ped}.pth' )
        # Utils.print_banner(f'Load lon model from {model_path}.')
        # Utils.print_banner(f'Load lat model from {model_path_lat}.')
        # Utils.print_banner(f'Load ita model from {model_path_ita}.')
        # Utils.print_banner(f'Load ped model from {model_path_ped}.')
        assert os.path.exists(model_path), print(model_path)
        assert os.path.exists(model_path), print(model_path_lat)
        ckpt = torch.load(model_path, map_location=self.device)
        ckpt_lat = torch.load(model_path_lat, map_location=self.device)
        ckpt_ita = torch.load(model_path_ita, map_location=self.device)

        ckpt_ped = torch.load(model_path_ped, map_location=self.device)
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

            self.value_net_ped.load_state_dict(ckpt_ped['value_net'])
            self.target_net_ped.load_state_dict(ckpt_ped['target_net'])
            self.optimizer_ped.load_state_dict(ckpt_ped['optimizer'])
# 对抗模型管理类
class ModelManager():
    def __init__(self,args, output_path, model = 'dqn'): 
        self.model= Model(args,output_path,model= model)
        self.features_range = {'x': [-60, 60], 'y': [-60, 60], 'vx': [-30, 30], 'vy': [-30, 30]}
    def observe(self):
        pass
    def normalize_obs(self,x, x_name = 'x'):
        return  utils.lmap(x, [self.features_range[x_name][0], self.features_range[x_name][1]], [-1, 1])
    # state
    def get_action(state):
        pass
