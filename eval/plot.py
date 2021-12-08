import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import indices


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
    folders = os.listdir(path)
    print("Reading folders from: ", path)
    print("Folders: ", folders)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        results = []
        files = os.listdir(folder_path)
        files = [f for f in files if f.endswith("_results.json")]
        for f in files:
            filepath = os.path.join(folder_path, f)
            with open(filepath, "r") as inf:
                results.append(json.load(inf))
        plt.figure(1)
        means, stdevs, mins, maxs = get_stats([x["train_loss"] for x in results])
        plot(means, stdevs, mins, maxs, "Training Loss", folder, "upper right")
        plt.figure(2)
        means, stdevs, mins, maxs = get_stats([x["test_loss"] for x in results])
        plot(means, stdevs, mins, maxs, "Testing Loss", folder, "upper right")
        plt.figure(3)
        means, stdevs, mins, maxs = get_stats([x["test_acc"] for x in results])
        plot(means, stdevs, mins, maxs, "Testing Accuracy", folder, "lower right")

    plt.figure(1)
    plt.savefig(os.path.join(path, "train_loss.png"))
    plt.figure(2)
    plt.savefig(os.path.join(path, "test_loss.png"))
    plt.figure(3)
    plt.savefig(os.path.join(path, "test_acc.png"))


def plot_parameters(path):
    plt.figure(4)
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        files = [f for f in files if f.endswith("_shared_params.json")]
        for f in files:
            filepath = os.path.join(folder_path, f)
            print("Working with ", filepath)
            with open(filepath, "r") as inf:
                loaded_dict = json.load(inf)
                del loaded_dict["order"]
                del loaded_dict["shapes"]
            assert len(loaded_dict["0"]) > 0
            assert "0" in loaded_dict.keys()
            counts = np.zeros(len(loaded_dict["0"]))
            for key in loaded_dict.keys():
                indices = np.array(loaded_dict[key])
                counts = np.pad(
                    counts,
                    max(np.max(indices) - counts.shape[0], 0),
                    "constant",
                    constant_values=0,
                )
                counts[indices] += 1
            plt.plot(np.arange(0, counts.shape[0]), counts, ".")
        print("Saving scatterplot")
        plt.savefig(os.path.join(folder_path, "shared_params.png"))


if __name__ == "__main__":
    assert len(sys.argv) == 2
    plot_results(sys.argv[1])
    plot_parameters(sys.argv[1])
