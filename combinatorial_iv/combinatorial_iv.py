import numpy as np
import math
from tqdm import tqdm
from joblib import Parallel, delayed  # 用于并行化
import argparse
import psutil

def generate_bernoulli_matrix(m, n, p):
    """ 生成 m×n 的伯努利随机矩阵（元素为0或1） """
    return np.random.binomial(1, p, size=(m, n))

def modify_array(arr):
    # 创建原数组的副本
    arr_copy = arr.copy()
    # 将副本中大于0的元素设为1，小于0的设为0
    arr_copy[arr_copy > 0] = 1
    arr_copy[arr_copy < 0] = 0
    return arr_copy

def select_sample(samples, indices, size):
    if len(samples) < size:
        return indices
    else:
        # 获取前 select_size 个最大的数及其索引
        top_indices = np.argsort(samples)[-size:][::-1]  # 排序并取出最大值索引

        # 从 non_zero_indices 中获取对应的值
        top_non_zero_indices = indices[top_indices]
        return top_non_zero_indices  # 返回可能为阳性样本的索引


def simulation_for_single_k_p_LOD(k, n, p, LOD_list, simu_num):
    """
    对于每次模拟，先生成完整的 n×n 的 A_full 以及信号 x，
    然后对每个 LOD，按均匀采样间隔 step=int(math.ceil(np.log(n))) 累积更新：
    每次 m 增加时，仅更新新增的行对 final_status 的影响，
    并记录采样时刻的检测总次数 num_tests = m + num_retest_samples 及敏感性。
    """
    results = []
    # 设置采样间隔
    step = int(math.ceil(np.log(n)))

    # 预计算全局记录点（所有LOD共享相同m_values）
    m_values = [m for m in range(1, n + 1) if m % step == 0 or m == n]

    for simu in tqdm(range(1, simu_num + 1)):
        # 生成信号 x（长度 n）：随机选取 k 个正样本，浓度随机取自 [100, 10^6]
        x = np.zeros(n)
        supp = np.random.choice(n, k, replace=False)
        x[supp] = np.random.randint(100, 10 ** 6 + 1, size=k)

        # 生成最大尺寸的混合矩阵 A_full（尺寸 n x n）并归一化每一行（防止除零）
        A_full = generate_bernoulli_matrix(n, n, p)
        # 对 A_full 的每一行归一化，避免除零
        row_sums = A_full.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        A_full_normalized = A_full / row_sums

        # 计算混合样本浓度 y = A_normalized @ x
        y_full = A_full_normalized @ x  # 结果为n维向量

        # 预计算所有LOD的阈值处理结果（内存允许情况下）
        y_processed_all = np.where(y_full[:, None] >= LOD_list, y_full[:, None], 0)  # (n, len(LOD_list))

        # 初始化每个LOD的状态矩阵（二维数组）
        final_status_all_lod = np.ones((len(LOD_list), n), dtype=int)

        # 修改结果容器结构
        simu_result = {
            "simulation": simu,
            "LOD_data": {}  # 嵌套存储所有LOD的结果
        }

        for lod in LOD_list:
            simu_result["LOD_data"][lod] = {
                "sensitivity_k": [],
                "sensitivity_2k": []
            }

        current_mix_num = np.zeros(n, dtype=int)

        for m_current in range(1, n + 1):
            idx = m_current - 1  # 当前行索引

            # 获取当前m对所有LOD的检测结果 (shape: (len(LOD_list),))
            current_y_processed = y_processed_all[idx, :]

            # 获取当前m下每个样本的混合次数
            current_mix_num += A_full[idx, :]

            # 向量化处理需要更新的LOD
            needs_update = (current_y_processed == 0)
            if np.any(needs_update):
                new_row = A_full_normalized[idx, :]  # shape: (n,)

                # 使用布尔索引批量更新需要修改的LOD状态
                # final_status_all_lod[needs_update] *= (new_row == 0)
                # 等效但更高效的写法：
                final_status_all_lod[needs_update] = np.logical_and(
                    final_status_all_lod[needs_update],  # 0表示样本已排除，1表示样本未排除
                    (new_row == 0),  # new_row中0元素对应该样本未参与混合
                )  # final_status_all_lod的大小为LOD*n,记录了某一LOD下n个样本中需要逐一检测的样本

            # 仅在记录点保存结果
            if m_current in m_values:
                for i in range(len(LOD_list)):
                    non_zero_indices = np.where(final_status_all_lod[i] != 0)
                    final_samples_all_lod = current_mix_num[non_zero_indices]  # 只给出未被排除样本的混合次数
                    top_k_indices = select_sample(final_samples_all_lod, non_zero_indices[0], k)
                    top_2k_indices = select_sample(final_samples_all_lod, non_zero_indices[0], 2*k)

                    common_k_indices = np.intersect1d(supp, top_k_indices)
                    common_2k_indices = np.intersect1d(supp, top_2k_indices)

                    sensitivity_k = np.where(k != 0, len(common_k_indices) / k, 0.0)
                    sensitivity_2k = np.where(k != 0, len(common_2k_indices) / k, 0.0)

                    simu_result["LOD_data"][LOD_list[i]]["sensitivity_k"].append(sensitivity_k)
                    simu_result["LOD_data"][LOD_list[i]]["sensitivity_2k"].append(sensitivity_2k)

        # 将完整模拟结果追加到列表
        results.append(simu_result)

    return (p, m_values, results)


def run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num):
    # 动态计算安全的并行进程数
    physical_cores = psutil.cpu_count(logical=False)
    n_jobs_main = max(1, physical_cores // 2)  # 保守设置

    all_results = Parallel(n_jobs=n_jobs_main, pre_dispatch="2*n_jobs", max_nbytes=None)(delayed(simulation_for_single_k_p_LOD)(k, n, p, LOD_list, simu_num)
                                       for p in tqdm(p_values))
    return all_results

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Search optimal m for given k_ratio')
    parser.add_argument('--k_ratio', type=float, required=True, help='k/n ratio ')
    parser.add_argument('--n', type=int, required=True, help='total number of items (must be a positive integer)')
    args = parser.parse_args()

    n = args.n  # n取5000,10000
    k = int(args.k_ratio * n)  # k_ratios = k/n 的比例 [0.0002, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
    p_values = np.array([2 ** (-i) for i in range(10, 0, -1)])  # p的取值从1/1024到1/2
    simu_num = 200
    LOD_list = np.array([1, 10, 50, 100, 500, 1000])
    # 运行模拟
    simulation_results = run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num)
    # 保存 simulation_results 到 npy 文件中
    np.save("/remote-home/share/iot_lijianing/Mix_Detection/results/combinatorial_iv/combinatorial_iv_data_n{}/n{}_kratio{}.npy".format(n, n, args.k_ratio), simulation_results)
    print("success")
