import numpy as np
import math
from tqdm import tqdm
from joblib import Parallel, delayed 
import argparse
import psutil

def generate_bernoulli_matrix(m, n, p):
    return np.random.binomial(1, p, size=(m, n))

def modify_array(arr):
    arr_copy = arr.copy()
    arr_copy[arr_copy > 0] = 1
    arr_copy[arr_copy < 0] = 0
    return arr_copy

def select_sample(samples, indices, size):
    if len(samples) < size:
        return indices
    else:
        top_indices = np.argsort(samples)[-size:][::-1] 
        top_non_zero_indices = indices[top_indices]
        return top_non_zero_indices  


def simulation_for_single_k_p_LOD(k, n, p, LOD_list, simu_num):
    results = []
    step = int(math.ceil(np.log(n)))

    m_values = [m for m in range(1, n + 1) if m % step == 0 or m == n]

    for simu in tqdm(range(1, simu_num + 1)):
        x = np.zeros(n)
        supp = np.random.choice(n, k, replace=False)
        x[supp] = np.random.randint(100, 10 ** 6 + 1, size=k)

        A_full = generate_bernoulli_matrix(n, n, p)
        row_sums = A_full.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        A_full_normalized = A_full / row_sums

        y_full = A_full_normalized @ x  

        y_processed_all = np.where(y_full[:, None] >= LOD_list, y_full[:, None], 0)  # (n, len(LOD_list))

        final_status_all_lod = np.ones((len(LOD_list), n), dtype=int)

        simu_result = {
            "simulation": simu,
            "LOD_data": {} 
        }

        for lod in LOD_list:
            simu_result["LOD_data"][lod] = {
                "sensitivity_k": [],
                "sensitivity_2k": []
            }

        current_mix_num = np.zeros(n, dtype=int)

        for m_current in range(1, n + 1):
            idx = m_current - 1  

            current_y_processed = y_processed_all[idx, :]

            current_mix_num += A_full[idx, :]

            needs_update = (current_y_processed == 0)
            if np.any(needs_update):
                new_row = A_full_normalized[idx, :]  # shape: (n,)

                final_status_all_lod[needs_update] = np.logical_and(
                    final_status_all_lod[needs_update],  
                    (new_row == 0),  
                )  

            if m_current in m_values:
                for i in range(len(LOD_list)):
                    non_zero_indices = np.where(final_status_all_lod[i] != 0)
                    final_samples_all_lod = current_mix_num[non_zero_indices]  
                    top_k_indices = select_sample(final_samples_all_lod, non_zero_indices[0], k)
                    top_2k_indices = select_sample(final_samples_all_lod, non_zero_indices[0], 2*k)

                    common_k_indices = np.intersect1d(supp, top_k_indices)
                    common_2k_indices = np.intersect1d(supp, top_2k_indices)

                    sensitivity_k = np.where(k != 0, len(common_k_indices) / k, 0.0)
                    sensitivity_2k = np.where(k != 0, len(common_2k_indices) / k, 0.0)

                    simu_result["LOD_data"][LOD_list[i]]["sensitivity_k"].append(sensitivity_k)
                    simu_result["LOD_data"][LOD_list[i]]["sensitivity_2k"].append(sensitivity_2k)

        results.append(simu_result)

    return (p, m_values, results)


def run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num):
    physical_cores = psutil.cpu_count(logical=False)
    n_jobs_main = max(1, physical_cores // 2) 

    all_results = Parallel(n_jobs=n_jobs_main, pre_dispatch="2*n_jobs", max_nbytes=None)(delayed(simulation_for_single_k_p_LOD)(k, n, p, LOD_list, simu_num)
                                       for p in tqdm(p_values))
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search optimal m for given k_ratio')
    parser.add_argument('--k_ratio', type=float, required=True, help='k/n ratio ')
    parser.add_argument('--n', type=int, required=True, help='total number of items (must be a positive integer)')
    args = parser.parse_args()

    n = args.n  # n={5000, 10000}
    k = int(args.k_ratio * n)  # k_ratios = k/n 
    p_values = np.array([2 ** (-i) for i in range(10, 0, -1)])
    simu_num = 200
    LOD_list = np.array([1, 10, 50, 100, 500, 1000])
    simulation_results = run_simulation_with_increasing_m_parallel(k, n, p_values, LOD_list, simu_num)
    np.save("./Mix_Detection/results/combinatorial_iv/combinatorial_iv_data_n{}/n{}_kratio{}.npy".format(n, n, args.k_ratio), simulation_results)
    print("success")
