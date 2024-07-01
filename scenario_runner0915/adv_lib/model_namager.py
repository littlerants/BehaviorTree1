
import os
import torch
from gym_sumo.gym_sumo import utils
from models.dqn_net import Net
import onnx
import onnxruntime
import numpy as np

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
        # self.init_networkonnx(model_type='onnx')
    def init_networkonnx(self,model_type='pth'):
        for key, val in self.args.model_config_list.items():
            if val.file_name:
                ab_model_path =  os.path.join(self.model_path, f'{val.file_name}.'+ model_type)
                print("----------------------------------------------------------")
                print("Load model ",key,"  from  ",ab_model_path)
                # if key == 'model_dynamic_wall':
                self.model_list[key] = onnxruntime.InferenceSession(ab_model_path)
                    # inputs = [-15.910923344951772, 0.5955110401160854, 0.5211584539665901, -5.800840889417668, 10.669737649833216, 1.5855170828783638, 0, 1, 1, 8.918071727460225, -1.1240416009932188, 3.051009648126073, 0.3604269637422266, 3.603269869645095, 1.5851837882679405, 1.0]
                    # pre = self.model_list[key].run(None, {self.model_list[key].get_inputs()[0].name: np.array(
                    #     inputs).astype(np.float32).reshape(1,16)    })
                    # print("pre:",pre)
                # ort_inputs = {self.model_list[key].get_inputs()[0].name: }
                # ort_outs = self.model_list[key].run(None, ort_inputs)
                # onnx_path = ab_model_path.replace("pth", "onnx")
                # self.export_to_onnx(onnx_path, val,self.model_list[key])
                # self.checmodel(onnx_path)

    def init_network(self):
        """
        init_network方法用于初始化网络。根据args.update_mode的不同取值，创建了不同的网络模型（Net或DuelingDQN），
        并将网络模型和优化器存储在self.value_net、self.target_net和self.optimizer中。
        """
        for key, val in self.args.model_config_list.items():
            if val.depart:
                pass
            else:
                self.model_list[key] = Net( val.state_dim,len(val.actions),self.args.model_config_list[key].hidden_size )
            if val.file_name:
                ab_model_path =  os.path.join(self.model_path, f'{val.file_name}.pth')    
                print("Load model ",key,"  from  ",ab_model_path)
                ckpt = torch.load(ab_model_path, map_location=self.device)
                self.model_list[key].load_state_dict(ckpt['value_net'])
                print("----------------------------------------------------------")
                onnx_path = ab_model_path.replace("pth", "onnx")
                self.export_to_onnx(onnx_path, val,self.model_list[key])
                self.checmodel(onnx_path)

    def export_to_onnx(self, path=None,value=None,net = None):
        net.eval()
        dummpy_inpute = torch.randn(1, value.config['state_dim'])
        torch.onnx.export(
            net,
            dummpy_inpute,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    def checmodel(self,onnx_model):
    # 我们可以使用异常处理的方法进行检验
        try:
            # 当我们的模型不可用时，将会报出异常
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
        else:
            # 模型可用时，将不会报出异常，并会输出“The model is valid!”
            print("The model is valid!")
# 对抗模型管理类
class ModelManager():
    def __init__(self,args, model_path, model = 'dqn'):
        self.model_config_list = args.model_config_list
        self.model= Model(args, model_path,model= model)
        self.features_range = {'x': [-60, 60], 'y': [-60, 60], 'vx': [-30, 30], 'vy': [-30, 30]}
    def observe(self):
        pass
    def normalize_obs(self,x, x_name = 'x'):
        return  utils.lmap(x, [self.features_range[x_name][0], self.features_range[x_name][1]], [-1, 1])
    # state
    def get_action(state):
        pass





if __name__ == "__main__":
    from model_namager import ModelManager
    from config import CONFIG
    args = CONFIG()
    model_manager = ModelManager(args, args.model_path)


