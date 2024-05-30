import torch.nn as nn
import torch
from torch.distributions import Normal
from Utils.math import *
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import scipy.stats as st
# st.norm.pdf()


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'mish':# nn.Mish()
            self.activation = torch.nn.functional.mish

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        # 待修正为通过网络预测
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (tensor): 输入张量

        Returns:
            tuple: 包含动作均值、动作对数标准差和动作标准差的元组
        """
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = torch.tanh(self.action_mean(x)).unsqueeze(0)

        action_log_std = self.action_log_std.expand_as(action_mean)

        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        """
        随机选择动作

        Args:
            x (tensor): 输入张量

        Returns:
            tuple: 包含动作、动作均值、动作标准差和动作对数概率的元组
        """
        action_mean, _, action_std = self.forward(x)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action, action_mean, action_std, action_log_prob

    def get_kl(self, x):
        """
        计算KL散度

        Args:
            x (tensor): 输入张量

        Returns:
            tensor: KL散度张量
        """
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()

        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        """
        计算动作的对数概率

        Args:
            x (tensor):输入张量
            action (tensor):选择的动作

        Returns:
            tensor: 对数概率张量
        """
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean.squeeze(), action_log_std.squeeze(), action_std.squeeze())

    def get_fim(self, x):
        """
        获取Fisher信息矩阵

        Args:
            x (tensor): 输入张量

        Returns:
            tuple: Fisher信息矩阵逆矩阵、动作均值和包含'dtd_id'和'std_index'的字典
        """
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


if __name__ == "__main__":
    import time
    input = torch.randn((1, 10))
    print(input.size())
    t0 = time.time()
    policy = Policy(state_dim=10, action_dim=2)
    print(time.time()-t0)
    # 分析FLOPs
    flops = FlopCountAnalysis(policy, input)
    print("FLOPs: ", flops.total())
    print(parameter_count_table(policy))
    total = sum([param.nelement() for param in policy.parameters()])
    print("Number of parameter: %.2fk" % (total / 1e3))

    action, action_mean, action_std, action_log_prob = policy.select_action(input)
    print(policy.get_log_prob(input, action))