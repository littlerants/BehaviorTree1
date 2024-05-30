import numpy as np
import math

# 计算两个高斯分布的kl散度
def kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    term1 = term2 = None
    try:
        term1 = np.log(sigma_q / sigma_p)
        term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2) - 0.5
    except Exception as e:
        term1 = np.log(sigma_q / (sigma_p + 1e-10))
        term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2 + 1e-10) - 0.5

    return term1 + term2


def get_similarity(expert_data: np.ndarray, virtual_data: np.ndarray):
    max_v = max(np.max(expert_data), np.max(virtual_data))
    min_v = min(np.min(expert_data), np.min(virtual_data))
    right = math.ceil(max_v)
    left = math.floor(min_v)
    ls = int(np.clip(right-left, 4, 10))
    sample_hist = np.histogram(np.array(expert_data), bins=np.linspace(left, right, ls), density=True)
    train_data_hist = np.histogram(virtual_data, bins=np.linspace(left, right, ls), density=True)

    intersection, union = 0.0, 0.0
    for i in range(sample_hist[0].shape[0]):
        intersection += min(sample_hist[0][i], train_data_hist[0][i])
        union += max(sample_hist[0][i], train_data_hist[0][i])

    return intersection / union


def evaluation_index(
             expert_data:np.ndarray,
             virtual_data: np.ndarray):
    exp_mean = np.mean(expert_data)
    vir_mean = np.mean(virtual_data)

    exp_std = np.std(expert_data)
    vir_std = np.std(virtual_data)
    # 计算专家数据与生成数据分布的kl散度
    kl = kl_divergence(exp_mean, exp_std, vir_mean, vir_std)
    # 计算专家数据与生成数据相似度
    # eps = 0 if abs(exp_mean - vir_mean) > 1e-10 else 1e-10
    # similarity = (1 - abs((exp_mean - vir_mean) / (exp_mean + eps)))
    similarity = get_similarity(expert_data, virtual_data)

    return {
        'kl': kl,
        'similarity': similarity,
    }