import numpy as np
import math
from tqdm import tqdm
from joblib import Parallel, delayed  # 用于并行化
import argparse
import psutil
import os

# 矩阵的生成方式:P固定,研究不同m下的success rate
def generate_binary_matrix(m, n, P):
    """ 生成 m×n 的行稀疏随机矩阵（元素为0或1) """
    matrix = np.zeros((m, n), dtype=int)
    for j in range(n):
        rows = np.random.choice(m, size=P, replace=True)
        matrix[rows, j] = 1
    return matrix

def select_sample(scores, indices, size):
    """
    从候选样本中选择得分最高的样本

    参数:
    scores: 候选样本的得分数组
    indices: 候选样本的原始索引数组
    size: 需要选择的样本数量

    返回:
    选中的样本原始索引数组
    """
    # 如果候选样本数量小于等于需要选择的数量，返回所有候选样本
    if len(scores) <= size:
        return indices

    # 获取得分最高的size个样本的索引（在候选样本中的位置）
    top_indices = np.argsort(scores)[-size:]

    # 转换为原始样本索引
    selected_indices = indices[top_indices]

    return selected_indices


def simulation_for_single_k_p_LOD(k, n, p, LOD_list, simu_num):
    # 计算固定P值（基于p和n）
    P = max(1, int(p * n))

    # 设置采样间隔
    if n == 5000:
        step = int(3 * math.ceil(np.log(n)))
    if n == 10000:
        step = int(6 * math.ceil(np.log(n)))
    # 预计算全局记录点（所有LOD共享相同m_values）
    m_values = [m for m in range(1, n + 1) if m % step == 0 or m == n]
    m_values = [m for m in m_values if m > P]  # 过滤掉m ≤ P的情况

    m_simu_result = [{
                "simulation": simu,
                "LOD_data": {lod: {"sensitivity_k": [], "sensitivity_2k": []} for lod in LOD_list}
            } for simu in range(1, simu_num + 1)]  # 列表中每个元素都是字典，每个字典对应一个simu_result

    skip_small_m = False  # 用于标记是否跳过较小的m的仿真
    zero_history = []  # 记录每个m值是否全为0的历史记录

    for m_current in tqdm(m_values[::-1]):
        # 如果需要跳过较小的m，则跳过
        if skip_small_m:
            # 填充所有数据为0
            for simu in range(1, simu_num + 1):
                for lod in LOD_list:
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_k"].append(0.0)
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_2k"].append(0.0)
            continue

        skip_all = False  # 每个m值重新初始化skip_all
        skip_list = []  # 每个m值重新初始化skip_list
        fill_value = 0.0  # 默认填充0

        for simu in range(1, simu_num + 1):
            # 如果已经确定跳过后续仿真（基于前100次结果）
            if skip_all and simu > 80:
                # 直接填充0并跳过计算
                for lod in LOD_list:
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_k"].append(fill_value)
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_2k"].append(fill_value)
                continue

            # 生成信号 x（长度 n）：随机选取k个正样本，浓度随机取自 [100, 10^6]
            x = np.zeros(n)
            supp = np.random.choice(n, k, replace=False)
            x[supp] = np.random.randint(100, 10 ** 6 + 1, size=k)

            if simu == 1 or simu % 40 == 0:
                A = generate_binary_matrix(m_current, n, P)  # 根据当前m生成新矩阵c
                row_sums = A.sum(axis=1)
                row_sums[row_sums == 0] = 1  # 避免除以零
                A_normalized = A / row_sums[:, np.newaxis]

            y = A_normalized @ x

            for lod_idx, lod in enumerate(LOD_list):
                # 1. 计算y中低于LOD的检测结果（阴性池）
                negative_pools = (y < lod)
                # 2. 找出所有阴性池中出现的样本（需要排除）
                # 获取所有阴性池的行
                negative_rows = A[negative_pools, :]
                # 计算需要排除的样本：在至少一个阴性池中出现的样本
                excluded_samples = np.any(negative_rows > 0, axis=0) if negative_rows.size > 0 else np.zeros(n, dtype=bool)

                # 3. 计算每个样本在阳性池中出现的次数（内积）
                # 阳性池标识
                positive_pools = (y >= lod)

                # 计算每个样本在阳性池中出现的次数
                scores = np.zeros(n)
                if np.any(positive_pools):
                    # 只考虑阳性池
                    positive_rows = A[positive_pools, :]
                    # 计算每个样本在阳性池中出现的次数
                    scores = np.sum(positive_rows, axis=0)

                # 4. 从未被排除的样本中选择top-k和top-2k
                candidate_indices = np.where(~excluded_samples)[0]
                candidate_scores = scores[candidate_indices]

                if len(candidate_indices) == 0:
                    # 如果没有候选样本，则敏感度为0
                    sensitivity_k = 0.0
                    sensitivity_2k = 0.0
                else:
                    top_k_indices = select_sample(candidate_scores, candidate_indices, k)
                    top_2k_indices = select_sample(candidate_scores, candidate_indices, 2*k)

                    common_k_indices = np.intersect1d(supp, top_k_indices)
                    common_2k_indices = np.intersect1d(supp, top_2k_indices)

                    sensitivity_k = np.where(k != 0, len(common_k_indices) / k, 0.0)
                    sensitivity_2k = np.where(k != 0, len(common_2k_indices) / k, 0.0)

                m_simu_result[simu-1]["LOD_data"][lod]["sensitivity_k"].append(sensitivity_k)
                m_simu_result[simu-1]["LOD_data"][lod]["sensitivity_2k"].append(sensitivity_2k)

                # 记录前100次仿真的结果
                if simu <= 80:
                    skip_list.append(sensitivity_k)
                    skip_list.append(sensitivity_2k)

            # 在第100次仿真后检查是否满足跳过条件
            if simu == 80:
                if all(v == 0.0 for v in skip_list):
                    skip_all = True
                    fill_value = 0.0
                if all(v == 1.0 for v in skip_list):
                    skip_all = True
                    fill_value = 1.0


        # 完成当前m的所有仿真后，检查这个m的所有结果是否全为0
        current_m_all_zero = all(
            simu_result["LOD_data"][lod]["sensitivity_k"][-1] == 0.0 and
            simu_result["LOD_data"][lod]["sensitivity_2k"][-1] == 0.0
            for simu_result in m_simu_result
            for lod in LOD_list
        )

        # 更新历史记录
        zero_history.append(current_m_all_zero)

        # 保持历史记录的长度不超过10
        zero_history = zero_history[-10:] if len(zero_history) > 10 else zero_history

        # 检查是否连续10个m值的结果都为0
        if len(zero_history) == 10 and all(zero_history):
            skip_small_m = True

    return (P, p, m_values, m_simu_result)

def consolidate_single_p_data(p_data, LOD_list):
    """
    对单个p值的模拟数据进行整合
    """
    P = p_data[0]
    p = p_data[1]
    m_values = p_data[2]
    results = p_data[3]

    consolidated_data = {}

    # 对每个LOD进行整合
    for lod in LOD_list:
        # 初始化存储所有仿真的中间结果
        all_lod_k = []
        all_lod_2k = []

        for simu_result in results:
            lod_data = simu_result["LOD_data"].get(lod)
            if lod_data is None:
                continue

            # sensitivity_k
            lod_data_k = np.array(lod_data["sensitivity_k"])
            lod_data_k[lod_data_k < 1] = 0  # 小于1的值设为0
            # sensitivity_2k
            lod_data_2k = np.array(lod_data["sensitivity_2k"])
            lod_data_2k[lod_data_2k < 1] = 0  # 小于1的值设为0

            # 收集处理后的数据
            all_lod_k.append(lod_data_k)
            all_lod_2k.append(lod_data_2k)

        # 计算平均值（axis=0表示沿仿真次数维度求平均）
        avg_lod_k = np.mean(all_lod_k, axis=0) if all_lod_k else np.zeros(len(m_values))
        avg_lod_2k = np.mean(all_lod_2k, axis=0) if all_lod_2k else np.zeros(len(m_values))

        # 保存该LOD的结果
        consolidated_data[lod] = {
            "sensitivity_k_avg": avg_lod_k[::-1],
            "sensitivity_2k_avg": avg_lod_2k[::-1]
        }

    return (p, P, m_values, consolidated_data)

def run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num):
    # 动态计算安全的并行进程数
    physical_cores = psutil.cpu_count(logical=False)
    n_jobs_main = max(1, physical_cores // 2)  # 保守设置

    # 先运行模拟获取原始数据
    print("Running simulations...")
    raw_results = Parallel(n_jobs=n_jobs_main, pre_dispatch="2*n_jobs", max_nbytes=None)(
        delayed(simulation_for_single_k_p_LOD)(k, n, p, LOD_list, simu_num)
        for p in tqdm(p_values)
    )

    # 然后对每个p值的数据进行整合
    print("Consolidating data...")
    consolidated_results = Parallel(n_jobs=n_jobs_main)(
        delayed(consolidate_single_p_data)(p_data, LOD_list)
        for p_data in tqdm(raw_results)
    )

    return consolidated_results


def generate_p_values(n):
    """根据n值生成对应的p_values"""
    if n == 5000:
        base_p = 1 / (2 ** 11)
    elif n == 10000:
        base_p = 1 / (2 ** 12)

    sqrt2 = np.sqrt(2)
    p_values = [base_p * (sqrt2 ** i) for i in range(10)]
    return p_values

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Search optimal m for given k_ratio')
    parser.add_argument('--k_ratio', type=float, required=True, help='k/n ratio ')
    parser.add_argument('--n', type=int, required=True, help='total number of items (must be a positive integer)')
    args = parser.parse_args()

    n = args.n  # n取5000,10000
    k = int(args.k_ratio * n)  # k_ratios = k/n 的比例 [0.0002, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    p_values = generate_p_values(n)
    simu_num = 200
    LOD_list = np.array([1, 10, 50, 100])
    # 运行模拟
    consolidated_results = run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num)

    # 为每个LOD创建单独的保存文件
    for lod in LOD_list:
        # 提取该LOD的数据
        lod_data = []
        for p, P, m_values, consolidated_dict in consolidated_results:
            if lod in consolidated_dict:
                lod_data.append((
                    p,
                    P,
                    m_values,
                    {
                        "sensitivity_k_avg": consolidated_dict[lod]["sensitivity_k_avg"],
                        "sensitivity_2k_avg": consolidated_dict[lod]["sensitivity_2k_avg"]
                    }
                ))

        # 创建保存目录
        save_dir = "/remote-home/share/iot_lijianing/Mix_Detection/results/combinatorial_iv_colnonfix_P/combinatorial_iv_colnonfix_P_data_consolidation_n{}_lod{}".format(
            n, lod)
        os.makedirs(save_dir, exist_ok=True)

        # 保存该LOD的数据
        save_path = os.path.join(save_dir, "n{}_kratio{}.npy".format(n, args.k_ratio))
        np.save(save_path, lod_data)
        print("Success! Consolidated data for LOD={} saved to: {}".format(lod, save_path))
