"""
Reproduce the figures of our paper
KSD Aggregated Goodness-of-fit Test
Antonin Schrab, Benjamin Guedj, Arthur Gretton
https://arxiv.org/pdf/2202.00824.pdf
Figures are generated using the dataframes in the 
directory results/ and are saved in the directory figures/.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import dataframe_image as dfi
from generate_data_nf import train_flow, create_multiscale_flow
from pathlib import Path

# matplotlib parameters
fs = 16
rcParams.update({"font.size": fs})
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

# create figures directory if it does not exist
Path("figures").mkdir(exist_ok=True, parents=True)

# Figure 1: Gamma experiment
results_gamma = pd.read_pickle("results/results_gamma.pkl")
power = results_gamma.unstack().to_numpy().T[[0, 3, 2, 1], :]
names = np.array(
    [
        r"\textsc{KSDAgg}",
        r"\textsc{KSD} median",
        r"\textsc{KSD} split",
        r"\textsc{KSD} split extra data",
    ]
)[[0, 3, 2, 1]]
lines = np.array(["-", "-", "-", "--"])[[0, 3, 2, 1]]
sigma = [0, 0.1, 0.2, 0.3, 0.4]
plt.figure(figsize=(6, 4))
for i in range(4):
    plt.plot(sigma, power[i], lines[i], label=names[i])
plt.legend()
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(-0.05, 1.05)
plt.xticks(sigma)
plt.xlabel("Shape parameter shift $s$", fontsize=fs)
plt.ylabel("Probability of rejecting $H_0$", labelpad=7, fontsize=fs)
plt.legend(
    fontsize=16,
    ncol=2,
    handleheight=0.5,
    labelspacing=0.4,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.48),
)
plt.savefig("figures/figure_1.png", dpi=300, bbox_inches="tight")
print("Figure 1 has been saved in figures/.")


# Figure 2: RBM experiment
results_rbm = pd.read_pickle("results/results_RBM.pkl")
power = results_rbm.unstack().to_numpy().T[[0, 3, 2, 1], :]
names = np.array(
    [
        r"\textsc{KSDAgg}",
        r"\textsc{KSD} median",
        r"\textsc{KSD} split",
        r"\textsc{KSD} split extra data",
    ]
)[[0, 3, 2, 1]]
lines = np.array(["-", "-", "-", "--"])[[0, 3, 2, 1]]
sigma = [0, 0.01, 0.02, 0.03]
plt.figure(figsize=(6, 4))
for i in range(4):
    plt.plot(sigma, power[i][:4], lines[i], label=names[i])
plt.xticks(sigma)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(-0.05, 1.05)
plt.xlabel("Perturbation standard deviation $\sigma$", fontsize=fs)
plt.ylabel("Probability of rejecting $H_0$", labelpad=7, fontsize=fs)
plt.legend(
    fontsize=16,
    ncol=2,
    handleheight=0.5,
    labelspacing=0.4,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.48),
)
plt.savefig("figures/figure_2.png", dpi=300, bbox_inches="tight")
print("Figure 2 has been saved in figures/.")


# Table 1: NF exeperiment level
results_nf = pd.read_pickle("results/results_NF_MNIST.pkl")
dfi.export(results_nf.rename(columns={"power/level": "level"}).loc["level"].unstack(), "figures/table_1.png")
print("Table 1 has been saved in figures/.")


# Figure 4: NF experiment power
results_nf = pd.read_pickle("results/results_NF_MNIST.pkl")
power = results_nf.loc["power"].unstack().to_numpy().T[[0, 3, 2, 1], :]
names = np.array(
    [
        r"\textsc{KSDAgg}",
        r"\textsc{KSD} median",
        r"\textsc{KSD} split",
        r"\textsc{KSD} split extra data",
    ]
)[[0, 3, 2, 1]]
lines = np.array(["-", "-", "-", "--"])[[0, 3, 2, 1]]
samples = [100, 200, 300, 400, 500]
plt.figure(figsize=(6, 4))
for i in range(4):
    plt.plot(samples, power[i], lines[i], label=names[i])
plt.legend()
plt.xticks(samples)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(-0.05, 1.05)
plt.xlabel("Sample size", fontsize=fs)
plt.ylabel("Probability of rejecting $H_0$", labelpad=7, fontsize=fs)
plt.legend(
    fontsize=16,
    ncol=2,
    handleheight=0.5,
    labelspacing=0.4,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.48),
)
plt.savefig("figures/figure_4.png", dpi=300, bbox_inches="tight")
print("Figure 4 has been saved in figures/.")


# For Figure 3: MNIST NF digits, we first need to load the MNIST dataset
# and pretrained Normalizing Flow model as in generate_data_nf.py
# This code is taken from Tutorial 11: Normalizing Flows for image modeling
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
import urllib.request
from urllib.error import HTTPError
import math
import os
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import torchvision

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "mnist/data_mnist"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "mnist/saved_models/tutorial11"
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial11/"
# Files to download
pretrained_files = ["MNISTFlow_multiscale.ckpt"]
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                e,
            )
# Convert images from 0-1 to 0-255 (integers)
def discretize(sample):
    return (sample * 255).to(torch.int32)


# Transformations applied on each image => make them a tensor and discretize
transform = transforms.Compose([transforms.ToTensor(), discretize])
# Loading the training dataset. We need to split it into a training and validation part
train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
# Load pretrained multiscale Normalizing Flow for MNIST
flow_dict = {"multiscale": {}}
flow_dict["multiscale"]["model"], flow_dict["multiscale"]["result"] = train_flow(
    create_multiscale_flow(), model_name="MNISTFlow_multiscale"
)

# Figure 3: MNIST NF digits
imgs_list = []
imgs_list.append([train_set[i][0] for i in range(4)])
pl.seed_everything(0)
imgs_list.append(
    flow_dict["multiscale"]["model"].sample(img_shape=[100, 8, 7, 7])[[0, 2, 9, 24]]
)
pl.seed_everything(0)
imgs_list.append(
    flow_dict["multiscale"]["model"].sample(img_shape=[100, 8, 7, 7])[[3, 6, 27, 5]]
)
filenames = ["figure_3_top", "figure_3_middle", "figure_3_bottom"]
plt.figure()
for i in range(len(imgs_list)):
    imgs = imgs_list[i]
    row_size = 4
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = (
        imgs.dtype == torch.int32
        if isinstance(imgs, torch.Tensor)
        else imgs[0].dtype == torch.int32
    )
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs / nrow))
    imgs = torchvision.utils.make_grid(
        imgs, nrow=nrow, pad_value=128 if is_int else 0.5
    )
    np_imgs = imgs.cpu().numpy()
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")
    plt.axis("off")
    plt.savefig("figures/" + filenames[i], dpi=300, bbox_inches="tight")
print("Figure 3 has been saved in figures/.")
