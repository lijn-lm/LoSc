import numpy as np
import math
from tqdm import tqdm
from joblib import Parallel, delayed  
import argparse
import psutil
import os

def generate_binary_matrix(m, n, P):
    matrix = np.zeros((m, n), dtype=int)
    for j in range(n):
        rows = np.random.choice(m, size=P, replace=True)
        matrix[rows, j] = 1
    return matrix

def select_sample(scores, indices, size):
    if len(scores) <= size:
        return indices

    top_indices = np.argsort(scores)[-size:]

    selected_indices = indices[top_indices]

    return selected_indices


def simulation_for_single_k_p_LOD(k, n, p, LOD_list, simu_num):
    P = max(1, int(p * n))

    if n == 5000:
        step = int(3 * math.ceil(np.log(n)))
    if n == 10000:
        step = int(6 * math.ceil(np.log(n)))
    m_values = [m for m in range(1, n + 1) if m % step == 0 or m == n]
    m_values = [m for m in m_values if m > P]  

    m_simu_result = [{
                "simulation": simu,
                "LOD_data": {lod: {"sensitivity_k": [], "sensitivity_2k": []} for lod in LOD_list}
            } for simu in range(1, simu_num + 1)]  

    skip_small_m = False  
    zero_history = [] 

    for m_current in tqdm(m_values[::-1]):
        if skip_small_m:
            for simu in range(1, simu_num + 1):
                for lod in LOD_list:
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_k"].append(0.0)
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_2k"].append(0.0)
            continue

        skip_all = False  
        skip_list = []  
        fill_value = 0.0 

        for simu in range(1, simu_num + 1):
            if skip_all and simu > 80:
                for lod in LOD_list:
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_k"].append(fill_value)
                    m_simu_result[simu - 1]["LOD_data"][lod]["sensitivity_2k"].append(fill_value)
                continue

            x = np.zeros(n)
            supp = np.random.choice(n, k, replace=False)
            x[supp] = np.random.randint(100, 10 ** 6 + 1, size=k)

            if simu == 1 or simu % 40 == 0:
                A = generate_binary_matrix(m_current, n, P)  
                row_sums = A.sum(axis=1)
                row_sums[row_sums == 0] = 1  
                A_normalized = A / row_sums[:, np.newaxis]

            y = A_normalized @ x

            for lod_idx, lod in enumerate(LOD_list):
                negative_pools = (y < lod)
                negative_rows = A[negative_pools, :]
                excluded_samples = np.any(negative_rows > 0, axis=0) if negative_rows.size > 0 else np.zeros(n, dtype=bool)

                positive_pools = (y >= lod)

                scores = np.zeros(n)
                if np.any(positive_pools):
                    positive_rows = A[positive_pools, :]
                    scores = np.sum(positive_rows, axis=0)

                candidate_indices = np.where(~excluded_samples)[0]
                candidate_scores = scores[candidate_indices]

                if len(candidate_indices) == 0:
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

                if simu <= 80:
                    skip_list.append(sensitivity_k)
                    skip_list.append(sensitivity_2k)

            if simu == 80:
                if all(v == 0.0 for v in skip_list):
                    skip_all = True
                    fill_value = 0.0
                if all(v == 1.0 for v in skip_list):
                    skip_all = True
                    fill_value = 1.0

        current_m_all_zero = all(
            simu_result["LOD_data"][lod]["sensitivity_k"][-1] == 0.0 and
            simu_result["LOD_data"][lod]["sensitivity_2k"][-1] == 0.0
            for simu_result in m_simu_result
            for lod in LOD_list
        )

        zero_history.append(current_m_all_zero)

        zero_history = zero_history[-10:] if len(zero_history) > 10 else zero_history

        if len(zero_history) == 10 and all(zero_history):
            skip_small_m = True

    return (P, p, m_values, m_simu_result)

def consolidate_single_p_data(p_data, LOD_list):
    P = p_data[0]
    p = p_data[1]
    m_values = p_data[2]
    results = p_data[3]

    consolidated_data = {}

    for lod in LOD_list:
        all_lod_k = []
        all_lod_2k = []

        for simu_result in results:
            lod_data = simu_result["LOD_data"].get(lod)
            if lod_data is None:
                continue

            # sensitivity_k
            lod_data_k = np.array(lod_data["sensitivity_k"])
            lod_data_k[lod_data_k < 1] = 0  
            # sensitivity_2k
            lod_data_2k = np.array(lod_data["sensitivity_2k"])
            lod_data_2k[lod_data_2k < 1] = 0 
   
            all_lod_k.append(lod_data_k)
            all_lod_2k.append(lod_data_2k)

        avg_lod_k = np.mean(all_lod_k, axis=0) if all_lod_k else np.zeros(len(m_values))
        avg_lod_2k = np.mean(all_lod_2k, axis=0) if all_lod_2k else np.zeros(len(m_values))

        consolidated_data[lod] = {
            "sensitivity_k_avg": avg_lod_k[::-1],
            "sensitivity_2k_avg": avg_lod_2k[::-1]
        }

    return (p, P, m_values, consolidated_data)

def run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num):
    physical_cores = psutil.cpu_count(logical=False)
    n_jobs_main = max(1, physical_cores // 2)  
    
    print("Running simulations...")
    raw_results = Parallel(n_jobs=n_jobs_main, pre_dispatch="2*n_jobs", max_nbytes=None)(
        delayed(simulation_for_single_k_p_LOD)(k, n, p, LOD_list, simu_num)
        for p in tqdm(p_values)
    )

    print("Consolidating data...")
    consolidated_results = Parallel(n_jobs=n_jobs_main)(
        delayed(consolidate_single_p_data)(p_data, LOD_list)
        for p_data in tqdm(raw_results)
    )

    return consolidated_results


def generate_p_values(n):
    if n == 5000:
        base_p = 1 / (2 ** 11)
    elif n == 10000:
        base_p = 1 / (2 ** 12)

    sqrt2 = np.sqrt(2)
    p_values = [base_p * (sqrt2 ** i) for i in range(10)]
    return p_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search optimal m for given k_ratio')
    parser.add_argument('--k_ratio', type=float, required=True, help='k/n ratio ')
    parser.add_argument('--n', type=int, required=True, help='total number of items (must be a positive integer)')
    args = parser.parse_args()

    n = args.n  # n={5000, 10000}
    k = int(args.k_ratio * n)  # k_ratios = k/n 
    p_values = generate_p_values(n)
    simu_num = 200
    LOD_list = np.array([1, 10, 50, 100])
    consolidated_results = run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num)

    for lod in LOD_list:
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

        save_dir = "./Mix_Detection/results/combinatorial_iv_colnonfix_P/combinatorial_iv_colnonfix_P_data_consolidation_n{}_lod{}".format(
            n, lod)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "n{}_kratio{}.npy".format(n, args.k_ratio))
        np.save(save_path, lod_data)
        print("Success! Consolidated data for LOD={} saved to: {}".format(lod, save_path))
