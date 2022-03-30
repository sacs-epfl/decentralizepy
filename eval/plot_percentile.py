import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch


def get_stats(l):
    assert len(l) > 0
    mean_dict, stdev_dict, min_dict, max_dict = {}, {}, {}, {}
    for key in l[0].keys():
        all_nodes = [i[key] for i in l]
        all_nodes = np.array(all_nodes)
        mean = np.mean(all_nodes)
        std = np.std(all_nodes)
        min = np.min(all_nodes)
        max = np.max(all_nodes)
        mean_dict[int(key)] = mean
        stdev_dict[int(key)] = std
        min_dict[int(key)] = min
        max_dict[int(key)] = max
    return mean_dict, stdev_dict, min_dict, max_dict


def plot(means, stdevs, mins, maxs, title, label, loc):
    plt.title(title)
    plt.xlabel("communication rounds")
    x_axis = list(means.keys())
    y_axis = list(means.values())
    err = list(stdevs.values())
    plt.errorbar(x_axis, y_axis, yerr=err, label=label)
    plt.legend(loc=loc)


def plot_results(path):
    """
    plots the percentiles
    Based on plot.py
    Parameters
    ----------
    path
        path to the folders from which to create the percentiles plots

    """
    folders = os.listdir(path)
    folders.sort()
    print("Reading folders from: ", path)
    print("Folders: ", folders)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        results = []
        all_shared_params = []
        machine_folders = os.listdir(folder_path)
        for machine_folder in machine_folders:
            mf_path = os.path.join(folder_path, machine_folder)
            if not os.path.isdir(mf_path):
                continue
            files = os.listdir(mf_path)
            shared_params = [f for f in files if f.endswith("_shared_parameters.json")]
            files = [f for f in files if f.endswith("_results.json")]
            for f in files:
                filepath = os.path.join(mf_path, f)
                with open(filepath, "r") as inf:
                    results.append(json.load(inf))
            for sp in shared_params:
                filepath = os.path.join(mf_path, sp)
                with open(filepath, "r") as spf:
                    all_shared_params.append(np.array(json.load(spf), dtype = np.int32))

        # Plot Training loss
        plt.figure(1)
        # Average of the shared parameters
        mean = np.mean(all_shared_params, axis=0)
        std = np.std(all_shared_params, axis=0)
        with open(
                os.path.join(path, "shared_params_avg_"+folder+".json"), "w"
        ) as mf:
            json.dump(mean.tolist(), mf)

        with open(
                os.path.join(path, "shared_params_std_"+folder+".json"), "w"
        ) as sf:
            json.dump(std.tolist(), sf)

        # copy jupyter notebook code
        percentile = np.percentile(mean, np.arange(0, 100, 1))
        plt.plot(np.arange(0, 100, 1), percentile, label=folder)
        plt.title('Shared parameters Percentiles')
        # plt.ylabel("Absolute frequency value")
        plt.xlabel("Percentiles")
        plt.xticks(np.arange(0, 110, 10))
        plt.legend(loc="lower right")

        plt.figure(2)
        sort = torch.sort(torch.tensor(mean)).values
        print(sort)
        length = sort.shape[0]
        length = int(length / 20)
        bins = [torch.sum(sort[length * i: length * (i + 1)]).item() for i in range(20)]
        total = np.sum(bins)
        perc = bins / total #np.divide(bins, total)
        print(perc)
        plt.bar(np.arange(0, 97.5, 5), perc, width=5, align='edge',
                label=folder)

        plt.title('Shared parameters Percentiles')
        # plt.ylabel("Absolute frequency value")
        plt.xlabel("Percentiles")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(path, f"percentiles_histogram_{folder}.png"), dpi=300)
        plt.clf()
        plt.cla()

    plt.figure(1)
    plt.savefig(os.path.join(path, "percentiles.png"), dpi=300)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    plot_results(sys.argv[1])