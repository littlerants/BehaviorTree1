import os
from gym_sumo.gym_sumo import utils
import onnxruntime
# 对抗模型初始化与加载类
class Model():
    def __init__(self,args,model_path,model = 'dqn') -> None:
        self.args = args
        self.model_path = model_path
        self.model = model
        self.train = False
        self.lr = 0.001
        self.model_list = {}
        # self.init_network()
        self.init_networkonnx(model_type='onnx')
    def init_networkonnx(self,model_type='pth'):
        for key, val in self.args.model_config_list.items():
            if val.file_name:
                ab_model_path =  os.path.join(self.model_path, f'{val.file_name}.'+ model_type)
                print("----------------------------------------------------------")
                print("Load model ",key,"  from  ",ab_model_path)
                # if key == 'model_dynamic_wall':
                self.model_list[key] = onnxruntime.InferenceSession(ab_model_path)

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


