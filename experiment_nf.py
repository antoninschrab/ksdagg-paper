"""
Run MNIST Normalizing Flow experiment using data 
from the directories data/NF_MNIST/ and parametric/NF_MNIST
as proposed in Section 4.5 of our paper
KSD Aggregated Goodness-of-fit Test
Antonin Schrab, Benjamin Guedj, Arthur Gretton
https://arxiv.org/pdf/2202.00824.pdf
Results are saved as dataframes in the directory results/.
"""

from kernel import stein_kernel_matrices, ratio_ksd_stdev
from ksd_single import ksd_parametric
from ksd_aggregated import ksdagg_parametric
from ksdagg import ksdagg
from pathlib import Path
import numpy as np
import pandas as pd
import time
import argparse

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

filenames = ["mnist_power", "mnist_level"]

# Computations (ksd_aggregated.py)
t = time.time()
verbose = True
weights_type = "uniform"
kernel_type = "imq"
beta_imq = 0.5
# B1 = 500 as in B1_parametric
# B2 = 500 as in B2_parametric
B3 = 50
d = 28**2
l_minus = -20
l_plus = 0
alpha = 0.05
repetitions = 200
for number_samples in [100, 200, 300, 400, 500]:
    for f in range(len(filenames)):
        filename = filenames[f]
        rs = np.random.RandomState(0)
        X_rep = np.load("data/NF_MNIST/testing/X_" + filename + ".npy").reshape(-1, d)
        score_X_rep = np.load(
            "data/NF_MNIST/testing/score_X_" + filename + ".npy"
        ).reshape(-1, d)
        B_parametric = np.load(
            "parametric/NF_MNIST/B_parametric" + str(number_samples) + ".npy"
        )
        B1_parametric = np.load(
            "parametric/NF_MNIST/B1_parametric" + str(number_samples) + ".npy"
        )
        B1_parametric_split = np.load(
            "parametric/NF_MNIST/B1_parametric_split" + str(number_samples) + ".npy"
        )
        B2_parametric = np.load(
            "parametric/NF_MNIST/B2_parametric" + str(number_samples) + ".npy"
        )
        median_bandwidth = np.load(
            "parametric/NF_MNIST/bandwidth" + str(number_samples) + ".npy"
        )
        if verbose:
            print(" ")
            print(
                "Starting n =",
                int(number_samples / 100),
                "/ 5 , f =",
                f + 1,
                "/",
                len(filenames),
            )
            print("bandwidth", median_bandwidth)
        ksdagg_results = np.zeros(repetitions)
        median_results = np.zeros(repetitions)
        split_results = np.zeros(repetitions)
        split_extra_data_results = np.zeros(repetitions)
        for r in range(repetitions):
            indices = rs.choice(X_rep.shape[0] - 1, size=number_samples, replace=False)
            X = X_rep[indices]
            score_X = score_X_rep[indices]
            X_extra = X_rep[indices + 1]
            score_X_extra = score_X_rep[indices + 1]

            # KSDAgg
            ksdagg_results[r] = ksdagg_parametric(
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
                    "Step: n = ",
                    int(number_samples / 100),
                    "/ 5 , f =",
                    f + 1,
                    "/",
                    len(filenames),
                    ",",
                    r + 1,
                    "/",
                    repetitions,
                    "time:",
                    time.time() - t,
                )
                t = time.time()
        power_level = (
            np.mean(ksdagg_results),
            np.mean(median_results),
            np.mean(split_results),
            np.mean(split_extra_data_results),
        )
        if verbose:
            for i in range(len(power_level)):
                print(filename[-5:], number_samples, test_names[i], power_level[i])

        for i in range(len(power_level)):
            index_vals.append((filename[-5:], number_samples, test_names[i]))
            results.append(power_level[i])

# save panda dataframe
index_names = (
    "null/alternative",
    "sample size",
    "test",
)
index = pd.MultiIndex.from_tuples(index_vals, names=index_names)
results_df = pd.Series(results, index=index).to_frame("power/level")
results_df.reset_index().to_csv("results/results_NF_MNIST.csv")
results_df.to_pickle("results/results_NF_MNIST.pkl")

# Computations (ksdagg.py) 
filenames = ["mnist_level", "mnist_power"]
repetitions = 200
for number_samples in [100, 200, 300, 400, 500]:
    for f in range(len(filenames)):
        filename = filenames[f]
        if verbose:
            print(" ")
            print(filename)
            print(number_samples)
        rs = np.random.RandomState(0)
        X_rep = np.load("data/NF_MNIST/testing/X_" + filename + ".npy").reshape(-1, 28 ** 2)
        score_X_rep = np.load(
            "data/NF_MNIST/testing/score_X_" + filename + ".npy"
        ).reshape(-1, 28 ** 2)
        B1_parametric = np.load(
            "parametric/NF_MNIST/B1_parametric_ksdagg_" + str(number_samples) + ".npy"
        )
        B2_parametric = np.load(
            "parametric/NF_MNIST/B2_parametric_ksdagg_" + str(number_samples) + ".npy"
        )
        bandwidths = np.load("parametric/NF_MNIST/bandwidths_ksdagg.npy")
        if verbose:
            print(
                "Starting n =",
                int(number_samples / 100),
                "/ 5 , f =",
                f + 1,
                "/",
                len(filenames),
            )
        ksdagg_results = []
        for r in range(repetitions):
            indices = rs.choice(X_rep.shape[0] - 1, size=number_samples, replace=False)
            X = X_rep[indices]
            score_X = score_X_rep[indices]

            # KSDAgg
            ksdagg_results.append(
                ksdagg(
                    X,
                    score_X,
                    bandwidths=bandwidths,
                    alpha=0.05,
                    kernel="imq",
                    weights_type="uniform", 
                    approx_type="parametric",
                    B1=B1_parametric,
                    B2=B2_parametric,
                    B3=50,
                    seed=42,
                    return_dictionary=False,
                )
            )
            
        power = np.mean(ksdagg_results)
        if verbose:
            print(power)
        np.save("results/ksdagg_" + filename + ".npy", power.reshape(1, -1))

if verbose:
    print("Dataframes for NF MNIST experiment have been saved in results/.")
