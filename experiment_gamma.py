"""
Run Gamma distribution experiment using data 
from the directory parametric/Gamma
as proposed in Section 4.3 of our paper
KSD Aggregated Goodness-of-fit Test
Antonin Schrab, Benjamin Guedj, Arthur Gretton
https://arxiv.org/pdf/2202.00824.pdf 
Results are saved as dataframes in the directory results/.
"""

from kernel import stein_kernel_matrices, ratio_ksd_stdev
from ksd_single import ksd_parametric
from ksd_aggregated import ksdagg_parametric
from ksdagg.np import ksdagg
from pathlib import Path
import numpy as np
import pandas as pd
import time
import argparse


def score_gamma(x, k, theta):
    return (k - 1) / x - 1 / theta


# create results directory if it does not exist
Path("results").mkdir(exist_ok=True, parents=True)

# panda dataframe: lists of indices and entries
index_vals = []
results = []

test_names = (
    "ksdagg",
    "median",
    "split",
    "split_extra_data",
)

# run all the experiments
t = time.time()
verbose = True
weights_type = "uniform"
kernel_type = "imq"
beta_imq = 0.5
# B1 = 500 as in B1_parametric
# B2 = 500 as in B2_parametric
B3 = 50
l_minus = 0
l_plus = 10
alpha = 0.05
number_samples = 500
perturbations = [0, 0.1, 0.2, 0.3, 0.4]
repetitions = 200
ksdagg_power = np.zeros(len(perturbations))
for s in range(len(perturbations)):
    perturbation = perturbations[s]
    k_p = 5
    k_q = k_p + perturbation
    theta_p = 5
    theta_q = theta_p
    rs = np.random.RandomState(s + 10)
    X_rep = rs.gamma(k_q, theta_q, (repetitions + 1, number_samples, 1))
    score_X_rep = score_gamma(X_rep, k_p, theta_p)
    B_parametric = np.load(
        "parametric/Gamma/B_parametric" + str(number_samples) + ".npy"
    )
    B1_parametric = np.load(
        "parametric/Gamma/B1_parametric" + str(number_samples) + ".npy"
    )
    B1_parametric_split = np.load(
        "parametric/Gamma/B1_parametric_split" + str(number_samples) + ".npy"
    )
    B2_parametric = np.load(
        "parametric/Gamma/B2_parametric" + str(number_samples) + ".npy"
    )
    median_bandwidth = np.load(
        "parametric/Gamma/bandwidth" + str(number_samples) + ".npy"
    )
    if verbose:
        print(" ")
        print("Starting s =", s + 1, "/", len(perturbations))
        print("perturbation", perturbation)
        print("bandwidth", median_bandwidth)
    ksdagg_param_results = np.zeros(repetitions)
    ksdagg_results = np.zeros(repetitions)
    median_results = np.zeros(repetitions)
    split_results = np.zeros(repetitions)
    split_extra_data_results = np.zeros(repetitions)
    for r in range(repetitions):
        X = X_rep[r]
        score_X = score_X_rep[r]
        X_extra = X_rep[r + 1]
        score_X_extra = score_X_rep[r + 1]
        
        # KSDAgg
        ksdagg_results[r] = ksdagg(
            X, 
            score_X,
            kernel="imq",
            number_bandwidths=10,
            weights_type="uniform", 
            approx_type="wild bootstrap",
            B1=2000, 
            B2=2000, 
            B3=50,
            seed=42,
        )

        # KSDAgg parametric
        ksdagg_param_results[r] = ksdagg_parametric(
            X,
            score_X,
            alpha,
            beta_imq,
            kernel_type,
            weights_type,
            l_minus,
            l_plus,
            median_bandwidth,
            B1_parametric,
            B2_parametric,
            B3,
        )

        # Median
        median_results[r] = ksd_parametric(
            X, score_X, alpha, beta_imq, kernel_type, median_bandwidth, B_parametric
        )

        # Stein kernel matrices
        bandwidths_collection = np.array(
            [2**i * median_bandwidth for i in range(l_minus, l_plus + 1)]
        )
        stein_kernel_matrices_list = stein_kernel_matrices(
            X,
            score_X,
            kernel_type,
            bandwidths_collection,
            beta_imq,
        )
        stein_kernel_matrices_list_extra_data = stein_kernel_matrices(
            X_extra,
            score_X_extra,
            kernel_type,
            bandwidths_collection,
            beta_imq,
        )

        # Split
        split_size = int(number_samples // 2)
        ratio_values = []
        for i in range(len(stein_kernel_matrices_list)):
            H = stein_kernel_matrices_list[i][:split_size, :split_size]
            ratio_values.append(ratio_ksd_stdev(H))
        selected_bandwidth = bandwidths_collection[np.argmax(ratio_values)]
        split_results[r] = ksd_parametric(
            X[split_size:],
            score_X[split_size:],
            alpha,
            beta_imq,
            kernel_type,
            selected_bandwidth,
            B1_parametric_split[np.argmax(ratio_values)],
        )

        # Split extra data
        ratio_values = []
        for i in range(len(stein_kernel_matrices_list_extra_data)):
            H_extra = stein_kernel_matrices_list_extra_data[i]
            ratio_values.append(ratio_ksd_stdev(H_extra))
        selected_bandwidth = bandwidths_collection[np.argmax(ratio_values)]
        split_extra_data_results[r] = ksd_parametric(
            X,
            score_X,
            alpha,
            beta_imq,
            kernel_type,
            selected_bandwidth,
            B1_parametric[np.argmax(ratio_values)],
        )

        if (r + 1) % 10 == 0 and verbose:
            print(
                "Step s =",
                s + 1,
                "/",
                len(perturbations),
                ",",
                r + 1,
                "/",
                repetitions,
                "time:",
                time.time() - t,
            )
            t = time.time()
    power_level = (
        np.mean(ksdagg_param_results),
        np.mean(median_results),
        np.mean(split_results),
        np.mean(split_extra_data_results),
    )
    ksdagg_power[s] = np.mean(ksdagg_results)
    if verbose:
        for i in range(len(power_level)):
            print(perturbation, test_names[i], power_level[i])

    for i in range(len(power_level)):
        index_vals.append((perturbation, test_names[i]))
        results.append(power_level[i])

# save panda dataframe
index_names = (
    "perturbation",
    "test",
)
index = pd.MultiIndex.from_tuples(index_vals, names=index_names)
results_df = pd.Series(results, index=index).to_frame("power/level")
results_df.reset_index().to_csv("results/results_gamma.csv")
results_df.to_pickle("results/results_gamma.pkl")
    
# save numpy array
np.save("results/ksdagg_gamma.npy", ksdagg_power.reshape(1, -1))

if verbose:
    print("Dataframes for Gamma experiment have been saved in results/.")

